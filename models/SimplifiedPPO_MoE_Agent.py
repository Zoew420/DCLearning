import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from datetime import datetime
import time
import traceback
from utils.logger import get_logger
from utils.auxiliary_loss import *
dataset_name = 'MIMIC-IV-MM'
logger = get_logger('SimplifiedPPO_MoE_Agent', dataset_name)

class ExperienceBuffer:
    """Simple experience buffer for PPO"""

    def __init__(self, device='cuda'):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.device = device
        self.logger = get_logger('ExperienceBuffer')

    def clear(self):
        """Clear the buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.logger.debug('Buffer cleared')

    def is_empty(self):
        """Check if buffer is empty"""
        return len(self.states) == 0

    def size(self):
        """Get buffer size"""
        return len(self.states)

    def __len__(self):
        """Get buffer size for len() function"""
        return len(self.states)

    def get_states(self):
        """Get all states from the buffer"""
        return self.states

    def get_actions(self):
        """Get all actions from the buffer"""
        return self.actions

    def get_rewards(self):
        """Get all rewards from the buffer"""
        return self.rewards

    def get_dones(self):
        """Get all done flags from the buffer"""
        return self.dones

    def get_log_probs(self):
        """Get all log probabilities from the buffer"""
        return self.log_probs

    def get_value_estimates(self):
        """Get all value estimates from the buffer"""
        return self.values

    def add(self, state, action, reward, done, log_prob, value=None):
        """Add an experience to the buffer"""
        try:
            is_batch = False
            batch_size = 1
            if 'img' in state and isinstance(state['img'], torch.Tensor):
                if state['img'].dim() > 1 and state['img'].size(0) > 1:
                    is_batch = True
                    batch_size = state['img'].size(0)
            if not is_batch:
                self._add_single_experience(state, action, reward, done, log_prob, value)
            else:
                if isinstance(action, torch.Tensor) and action.size(0) != batch_size:
                    self.logger.warning(f'动作批次大小 ({action.size(0)}) 与状态批次大小 ({batch_size}) 不匹配')
                    if action.size(0) == 1:
                        action = action.expand(batch_size, -1) if action.dim() > 1 else action.expand(batch_size)
                    elif action.size(0) > batch_size:
                        action = action[:batch_size]
                    else:
                        batch_size = action.size(0)
                for i in range(batch_size):
                    single_state = {}
                    for key, tensor in state.items():
                        if isinstance(tensor, torch.Tensor) and tensor.size(0) > i:
                            single_state[key] = tensor[i:i + 1]
                        else:
                            single_state[key] = tensor
                    single_action = action[i] if isinstance(action, torch.Tensor) and action.size(0) > i else action
                    single_reward = reward[i] if isinstance(reward, (list, tuple, torch.Tensor)) and len(reward) > i else reward
                    single_done = done[i] if isinstance(done, (list, tuple, torch.Tensor)) and len(done) > i else done
                    single_log_prob = log_prob[i] if isinstance(log_prob, torch.Tensor) and log_prob.size(0) > i else log_prob
                    single_value = value[i] if value is not None and isinstance(value, torch.Tensor) and (value.size(0) > i) else value
                    self._add_single_experience(single_state, single_action, single_reward, single_done, single_log_prob, single_value)
        except Exception as e:
            raise

    def _add_single_experience(self, state, action, reward, done, log_prob, value=None):
        """添加单个样本的经验到缓冲区"""
        self.states.append(state)
        self.actions.append(action)
        if isinstance(reward, torch.Tensor):
            reward = reward.item() if reward.numel() == 1 else float(reward)
        self.rewards.append(float(reward))
        if isinstance(done, torch.Tensor):
            done = bool(done.item()) if done.numel() == 1 else bool(done)
        self.dones.append(bool(done))
        self.log_probs.append(log_prob)
        if value is not None:
            if isinstance(value, torch.Tensor):
                value = value.detach()
            self.values.append(value)

class SimplifiedPPOTrainer:
    """Simplified PPO trainer that only updates the gate network"""

    def __init__(self, model, gate_optimizer, gamma=0.99, lam=0.95, clip_ratio=0.2, entropy_coef=0.05, value_coef=0.5, max_grad_norm=0.5, batch_size=64, ppo_epochs=3, ppo_buffer_size=2048, device='cuda'):
        self.model = model
        self.gate_optimizer = gate_optimizer
        self.device = device
        self.batch_size = batch_size
        self.buffer = ExperienceBuffer(device=device)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.ppo_buffer_size = ppo_buffer_size
        self.logger = get_logger('SimplifiedPPOTrainer')
        self.expert_usage = torch.zeros(7, device=device)
        self.total_steps = 0
        self.emergency_mode = False
        self.logger.info('Running with full PPO updates enabled')

    def compute_returns(self, rewards):
        """Compute simple returns with discount"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self):
        """Safely update the policy using PPO with robust error handling"""
        if self.buffer.size() == 0:
            self.logger.warning('Empty buffer, skipping update')
            return self._default_metrics()
        try:
            states = self.buffer.get_states()
            actions = self.buffer.get_actions()
            rewards = self.buffer.get_rewards()
            old_log_probs = self.buffer.get_log_probs()
            with torch.no_grad():
                if not isinstance(rewards, torch.Tensor):
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                else:
                    rewards_tensor = rewards.detach().clone()
                reward_mean = rewards_tensor.mean().item()
                reward_std = rewards_tensor.std(unbiased=False).item()
                returns = self.compute_returns(rewards_tensor)
                img_tensors = []
                text_tensors = []
                time_tensors = []
                for state in states:
                    if 'img' in state and isinstance(state['img'], torch.Tensor):
                        img_tensors.append(state['img'].detach().clone())
                    else:
                        self.logger.warning("Missing or invalid 'img' in state")
                        continue
                    if 'text' in state and isinstance(state['text'], torch.Tensor):
                        text_tensors.append(state['text'].detach().clone())
                    else:
                        self.logger.warning("Missing or invalid 'text' in state")
                        continue
                    if 'time' in state and isinstance(state['time'], torch.Tensor):
                        time_tensors.append(state['time'].detach().clone())
                    else:
                        self.logger.warning("Missing or invalid 'time' in state")
                        continue
                if not img_tensors or not text_tensors or (not time_tensors):
                    self.logger.error('No valid state data found in buffer')
                    self.buffer.clear()
                    return self._default_metrics()
                try:
                    img_tensor = torch.cat(img_tensors, dim=0)
                    text_tensor = torch.cat(text_tensors, dim=0)
                    time_tensor = torch.cat(time_tensors, dim=0)
                except Exception as e:
                    raise
                if not isinstance(actions, torch.Tensor):
                    try:
                        actions_list = []
                        for a in actions:
                            if isinstance(a, torch.Tensor):
                                if a.numel() == 1:
                                    actions_list.append(a.item())
                                else:
                                    actions_list.append(a[0].item() if a.dim() > 0 else a.item())
                            elif isinstance(a, (list, tuple)) and len(a) > 0:
                                actions_list.append(int(a[0]) if isinstance(a[0], (int, float)) else 0)
                            elif isinstance(a, (int, float)):
                                actions_list.append(int(a))
                            else:
                                self.logger.warning(f'Unknown action type: {type(a)}, using default value 0')
                                actions_list.append(0)
                        actions_tensor = torch.tensor(actions_list, dtype=torch.long, device=self.device)
                    except Exception as e:
                        raise
                else:
                    actions_tensor = actions.clone().detach().to(dtype=torch.long)
                if not isinstance(old_log_probs, torch.Tensor):
                    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
                else:
                    old_log_probs_tensor = old_log_probs.detach().clone()
            policy_loss_val = 0
            value_loss_val = 0
            entropy_val = 0
            total_loss_val = 0
            try:
                self.gate_optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    img_fresh = img_tensor.detach().clone()
                    text_fresh = text_tensor.detach().clone()
                    time_fresh = time_tensor.detach().clone()
                    actions_fresh = actions_tensor.detach().clone()
                with torch.set_grad_enabled(True):
                    action_probs, value_preds, expert_probs = self.model.evaluate_actions(img_fresh, text_fresh, time_fresh, actions_fresh)
                    if action_probs.dim() > old_log_probs_tensor.dim():
                        action_probs = action_probs.squeeze(-1)
                    if value_preds.dim() > returns.dim():
                        value_preds = value_preds.squeeze(-1)
                    if old_log_probs_tensor.dim() > action_probs.dim():
                        old_log_probs_tensor = old_log_probs_tensor.squeeze(-1)
                    elif old_log_probs_tensor.dim() < action_probs.dim():
                        old_log_probs_tensor = old_log_probs_tensor.unsqueeze(-1)
                    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-08)
                    advantages = returns - value_preds.detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-08)
                    entropy = -(expert_probs * torch.log(expert_probs + 1e-10)).sum(dim=1).mean()
                    new_log_probs = torch.log(action_probs + 1e-10)
                    ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value_preds, returns)
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.large_moe_model.ppo_gate.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.model.large_moe_model.value_network.parameters(), self.max_grad_norm)
                    self.gate_optimizer.step()
                    policy_loss_val = policy_loss.item()
                    value_loss_val = value_loss.item()
                    entropy_val = entropy.item()
                    total_loss_val = loss.item()
            except Exception as e:
                raise
            finally:
                self.buffer.clear()
                torch.cuda.empty_cache()
            return {'policy_loss': policy_loss_val, 'value_loss': value_loss_val, 'entropy': entropy_val, 'total_loss': total_loss_val, 'reward_mean': reward_mean, 'reward_std': reward_std}
        except Exception as e:
            raise

    def _default_metrics(self):
        """Return default metrics in case of error"""
        return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0, 'reward_mean': 0, 'reward_std': 0}

    def add_experience(self, state, action, reward, done, log_prob, value=None):
        """Add an experience to the buffer with size limit"""
        if self.buffer.size() >= self.ppo_buffer_size:
            self.logger.warning('Buffer reached size limit of {}, clearing oldest entries'.format(self.ppo_buffer_size))
            self.buffer.states = self.buffer.states[len(self.buffer.states) // 2:]
            self.buffer.actions = self.buffer.actions[len(self.buffer.actions) // 2:]
            self.buffer.rewards = self.buffer.rewards[len(self.buffer.rewards) // 2:]
            self.buffer.dones = self.buffer.dones[len(self.buffer.dones) // 2:]
            self.buffer.log_probs = self.buffer.log_probs[len(self.buffer.log_probs) // 2:]
            if len(self.buffer.values) > 0:
                self.buffer.values = self.buffer.values[len(self.buffer.values) // 2:]
        self.buffer.add(state, action, reward, done, log_prob, value)

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store a transition in the buffer, with clipped reward"""
        if isinstance(reward, torch.Tensor):
            reward = torch.clamp(reward, -10.0, 10.0)
        else:
            reward = np.clip(reward, -10.0, 10.0)
        self.add_experience(state, action, reward, done, log_prob, value)

    def clear_buffer(self):
        """Clear the buffer"""
        self.buffer.clear()

def compute_ppo_reward(args, large_feat, small_feat, task_loss, distillation_weight=0.5):
    """Compute reward for PPO gate training
    
    Args:
        large_feat: Features from the large model
        small_feat: Features from the small model 
        task_loss: Task loss value
        distillation_weight: Weight for distillation component
        
    Returns:
        Combined reward signal
    """
    reward_logger = get_logger('compute_ppo_reward')
    device = large_feat.device
    batch_size = max(large_feat.size(0), small_feat.size(0))
    if large_feat.size(0) != batch_size:
        if large_feat.size(0) == 1:
            large_feat = large_feat.expand(batch_size, -1)
        else:
            reward_logger.warning(f'large_feat batch size mismatch: {large_feat.size(0)} vs {batch_size}')
            large_feat = torch.zeros(batch_size, large_feat.size(1), device=device)
    if small_feat.size(0) != batch_size:
        if small_feat.size(0) == 1:
            small_feat = small_feat.expand(batch_size, -1)
        else:
            reward_logger.warning(f'small_feat batch size mismatch: {small_feat.size(0)} vs {batch_size}')
            small_feat = torch.zeros(batch_size, small_feat.size(1), device=device)
    if large_feat.size(1) != small_feat.size(1):
        if not hasattr(compute_ppo_reward, 'projection'):
            input_dim = min(large_feat.size(1), small_feat.size(1))
            output_dim = max(large_feat.size(1), small_feat.size(1))
            compute_ppo_reward.projection = torch.nn.Linear(input_dim, output_dim).to(device)
        if large_feat.size(1) < small_feat.size(1):
            large_feat = compute_ppo_reward.projection(large_feat)
        else:
            small_feat = compute_ppo_reward.projection(small_feat)
    large_norm = F.normalize(large_feat, dim=1)
    small_norm = F.normalize(small_feat, dim=1)
    kl_loss = compute_kl_distillation_loss(large_norm, small_norm)
    reward_negative = convert_loss_to_reward(kl_loss, method='negative')
    reward_inverse = convert_loss_to_reward(kl_loss, method='inverse')
    reward_exp = convert_loss_to_reward(kl_loss, method='exp')
    if args.reward_type == 'negative':
        similarity_reward = reward_negative
    elif args.reward_type == 'inverse':
        similarity_reward = reward_inverse
    elif args.reward_type == 'exp':
        similarity_reward = reward_exp
    normalized_task_loss = np.clip(task_loss, 0.0, 5.0) / 5.0
    task_reward = 1.0 - normalized_task_loss
    combined_reward = (1 - distillation_weight) * task_reward + distillation_weight * similarity_reward.mean()
    final_reward = combined_reward * 1.0 + 0.5
    if hasattr(compute_ppo_reward, 'call_counter'):
        compute_ppo_reward.call_counter += 1
    else:
        compute_ppo_reward.call_counter = 0
    if compute_ppo_reward.call_counter % 100 == 0:
        reward_logger.debug(f'Reward components - Task loss: {task_loss:.4f}, Task reward: {task_reward:.4f}, Similarity reward: {similarity_reward.mean():.4f}, Combined reward: {final_reward:.4f}')
    return final_reward
