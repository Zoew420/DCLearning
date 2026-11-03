import torch
import torch.nn as nn
import torch.optim as optim
from models.SimplifiedPPO_MoE_Agent import SimplifiedPPOTrainer, compute_ppo_reward
from utils.logger import get_logger
import torch.nn.functional as F
import logging
import traceback
dataset_name = 'MIMIC-IV-MM'
logger = get_logger('PPO_MultiModal', dataset_name)

class PPOMultiModalModel(nn.Module):
    """
    MultiModal model that uses PPO to learn modal relationships from a small model
    """

    def __init__(self, img_dim=1536, text_dim=5120, time_dim=1024, hidden_dim=512, num_modality=3, num_classes=2, cls_dropout=0.1, device='cuda', clip_ratio=0.2, gate_temperature=1.0, small_model_path=None):
        """初始化多模态模型

        Args:
            img_dim: 图像特征维度
            text_dim: 文本特征维度
            time_dim: 时间序列特征维度
            hidden_dim: 隐藏层维度
            num_modality: 模态数量
            num_classes: 分类类别数量
            cls_dropout: 分类器的dropout率
            device: 设备 ('cuda' 或 'cpu')
            gate_temperature: 门控网络softmax的温度参数
        """
        super(PPOMultiModalModel, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.img_feat_dim = img_dim
        self.text_feat_dim = text_dim
        self.time_feat_dim = time_dim
        self.num_experts = 7
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.dropout = cls_dropout
        self.large_moe_model = LargeMoEModelWithPPOGate(input_features=[self.img_feat_dim, self.text_feat_dim, self.time_feat_dim], hidden_dim=self.hidden_dim, device=self.device, gate_temperature=gate_temperature)
        classifier = nn.Linear(1536, num_classes)
        if small_model_path is not None:
            state = torch.load(small_model_path, map_location='cpu', weights_only=True)
            state = state.get('model_state_dict', state.get('state_dict', state))
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            head = {k.removeprefix('cls_head.'): v for k, v in state.items() if k.startswith('cls_head.')}
            if not head:
                head = {k.removeprefix('large_classifier.'): v for k, v in state.items() if k.startswith('large_classifier.')}
            if not head and set(state) == {'weight', 'bias'}:
                head = state
            classifier.load_state_dict(head, strict=True)
        self.fusion_weight = nn.Parameter(torch.ones(1, 2), requires_grad=True)
        self.fusion_weight.data.fill_(0.5)
        self.large_classifier = classifier
        self.for_cls_projection = nn.Linear(3 * self.hidden_dim, 1536)
        self.logger.info(f'Classifier input dim: {self.hidden_dim}, output dim: {num_classes}')
        self.ppo_trainer = None
        self.episode_data = []

    def set_ppo_optimizer(self, optimizer, batch_size=64, ppo_epochs=3, ppo_buffer_size=2048, clip_ratio=0.2, gamma=0.99, lam=0.95, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        """Set a dedicated optimizer for PPO updates with all necessary parameters
        
        This version only optimizes the gate network and value network in the MoE model
        """
        gate_params = list(self.large_moe_model.ppo_gate.parameters()) + list(self.large_moe_model.value_network.parameters())
        if optimizer is None:
            gate_optimizer = torch.optim.Adam(gate_params, lr=0.0003)
        else:
            gate_optimizer = optimizer
        self.ppo_trainer = SimplifiedPPOTrainer(model=self, gate_optimizer=gate_optimizer, batch_size=batch_size, ppo_epochs=ppo_epochs, ppo_buffer_size=ppo_buffer_size, clip_ratio=clip_ratio, gamma=gamma, lam=lam, entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm, device=self.device)
        self.logger.info('PPO optimizer set to train only the gate network and value network')

    def forward(self, image, text, time, small_feat=None, training=False, collect_policy_info=True):
        """Forward pass for both training and inference
        
        Args:
            image: Image features 
            text: Text features
            time: Time series features
            small_feat: Optional small model features (for distillation)
            training: Whether in training mode
            collect_policy_info: Whether to collect policy info (for PPO)
            
        Returns:
            logits: Classification logits
            features: Fused features
            policy_info: Policy information for PPO update
        """
        with torch.set_grad_enabled(training):
            try:
                if isinstance(image, torch.Tensor):
                    image_input = image.detach() if not training else image
                else:
                    image_input = image
                if isinstance(text, torch.Tensor):
                    text_input = text.detach() if not training else text
                else:
                    text_input = text
                if isinstance(time, torch.Tensor):
                    time_input = time.detach() if not training else time
                else:
                    time_input = time
                features, policy_info = self.large_moe_model(image_input, text_input, time_input, training=training, collect_policy_info=collect_policy_info)
                small_logits = self.large_classifier(small_feat)
                features = self.for_cls_projection(features)
                large_logits = self.large_classifier(features)
                fusion_weight = F.softmax(self.fusion_weight, dim=1)
                logits = fusion_weight[:, 0] * small_logits + fusion_weight[:, 1] * large_logits
                logits = large_logits
                if not collect_policy_info:
                    policy_info = None
                return (logits, features, policy_info)
            except Exception as e:
                raise

    def store_transition(self, state, action, reward, value=None, log_prob=None, done=False):
        """Store transition for PPO update"""
        if self.ppo_trainer is None:
            self.logger.warning('PPO trainer not initialized, skipping transition storage')
            return
        self.ppo_trainer.store_transition(state, action, reward, value, log_prob, done)

    def update_policy(self):
        """Update policy using PPO"""
        if self.ppo_trainer is None:
            self.logger.warning('PPO trainer not initialized, skipping policy update')
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0}
        metrics = self.ppo_trainer.update()
        return metrics

    def evaluate_actions(self, img, text, time, actions):
        """
        Evaluate actions for PPO update - delegates to the large_moe_model's evaluate_actions
        
        Args:
            img: Image features
            text: Text features
            time: Time features
            actions: Actions to evaluate
            
        Returns:
            action_probs: Action probabilities
            value_estimates: Value estimates
            expert_probs: Expert weights
        """
        if img is not None:
            img = img.to(self.device)
        if text is not None:
            text = text.to(self.device)
        if time is not None:
            time = time.to(self.device)
        if actions is not None and isinstance(actions, torch.Tensor):
            actions = actions.to(self.device)
        return self.large_moe_model.evaluate_actions(img, text, time, actions)

    def _reset_episode_data(self):
        """Reset episode-specific data"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        self.episode_dones = []

    def set_gate_temperature(self, temperature):
        """Set gate temperature for inference"""
        self.large_moe_model.set_gate_temperature(temperature)

    def get_gate_temperature(self):
        """Get current gate temperature"""
        return self.large_moe_model.get_gate_temperature()

class ModalityExpert(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LargeMoEModelWithPPOGate(nn.Module):
    """
    Large Mixture of Experts model with PPO-compatible gate network
    """

    def __init__(self, input_features, hidden_dim=512, dropout=0.1, device='cuda', gate_temperature=1.0):
        super(LargeMoEModelWithPPOGate, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.gate_temperature = gate_temperature
        self.logger = get_logger('LargeMoEModelWithPPOGate')
        self.img_projector = nn.Linear(input_features[0], hidden_dim)
        self.text_projector = nn.Linear(input_features[1], hidden_dim)
        self.time_projector = nn.Linear(input_features[2], hidden_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=3 * hidden_dim, num_heads=8, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(3 * hidden_dim)
        output_dim = 3 * hidden_dim
        self.experts = nn.ModuleList([ModalityExpert(hidden_dim, hidden_dim, output_dim), ModalityExpert(hidden_dim, hidden_dim, output_dim), ModalityExpert(hidden_dim, hidden_dim, output_dim), ModalityExpert(2 * hidden_dim, hidden_dim, output_dim), ModalityExpert(2 * hidden_dim, hidden_dim, output_dim), ModalityExpert(2 * hidden_dim, hidden_dim, output_dim), ModalityExpert(3 * hidden_dim, hidden_dim, output_dim)])
        self.ppo_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 7))
        self.value_network = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.register_buffer('expert_usage', torch.zeros(7, device=device))
        self.register_buffer('expert_usage_history', torch.zeros(7, device=device))
        self.total_steps = 0

    def get_diversity_bonus(self, expert_weights):
        """
        计算多样性奖励，鼓励选择少数几个性能好的专家
        """
        temperature = 0.5
        sharpened_weights = expert_weights ** (1 / temperature)
        sharpened_weights = sharpened_weights / (sharpened_weights.sum(dim=1, keepdim=True) + 1e-10)
        batch_avg_weights = sharpened_weights.mean(dim=0)
        self.expert_usage_history = 0.85 * self.expert_usage_history + 0.15 * batch_avg_weights
        sparsity_target = 3
        l1_penalty = torch.abs(expert_weights).sum(dim=1)
        eps = 1e-10
        entropy = -(expert_weights * torch.log(expert_weights + eps)).sum(dim=1)
        effective_experts = torch.exp(entropy)
        expert_count_penalty = 0.2 * torch.abs(effective_experts - sparsity_target)
        k = 3
        topk_weights, _ = torch.topk(expert_weights, k, dim=1)
        topk_entropy = -(topk_weights * torch.log(topk_weights + eps)).sum(dim=1)
        usage_penalty = torch.matmul(expert_weights, self.expert_usage_history.unsqueeze(1)).squeeze()
        diversity_bonus = 0.8 * topk_entropy - 0.3 * l1_penalty - 0.2 * usage_penalty - expert_count_penalty
        return diversity_bonus

    def forward(self, img, text, time, training=False, collect_policy_info=True):
        """Forward pass through MoE model with PPO gate"""
        try:
            img_feat = self.img_projector(img)
            text_feat = self.text_projector(text)
            time_feat = self.time_projector(time)
            expert_outputs = [self.experts[0](img_feat), self.experts[1](text_feat), self.experts[2](time_feat), self.experts[3](torch.cat([img_feat, text_feat], dim=1)), self.experts[4](torch.cat([img_feat, time_feat], dim=1)), self.experts[5](torch.cat([text_feat, time_feat], dim=1)), self.experts[6](torch.cat([img_feat, text_feat, time_feat], dim=1))]
            expert_outputs = torch.stack(expert_outputs, dim=1)
            gate_input = torch.cat([img_feat, text_feat, time_feat], dim=1)
            gate_logits = self.ppo_gate(gate_input)
            if training:
                temperature = max(1.0, self.gate_temperature)
            else:
                temperature = self.gate_temperature
            expert_probs = F.softmax(gate_logits / temperature, dim=1)
            value_estimate = self.value_network(gate_input)
            diversity_bonus = self.get_diversity_bonus(expert_probs).unsqueeze(1)
            if not training:
                if temperature < 0.1:
                    expert_idx = torch.argmax(gate_logits, dim=1, keepdim=True)
                    one_hot = torch.zeros_like(expert_probs)
                    one_hot.scatter_(1, expert_idx, 1.0)
                    expert_weights = one_hot
                else:
                    expert_weights = expert_probs
                combined_output = torch.einsum('be,beh->bh', expert_weights, expert_outputs)
                with torch.no_grad():
                    self.expert_usage += expert_weights.sum(dim=0).detach()
                    self.expert_usage_history = 0.99 * self.expert_usage_history + 0.01 * expert_weights.sum(dim=0).detach()
                    self.total_steps += expert_weights.size(0)
                if collect_policy_info:
                    policy_info = {'expert_probs': expert_probs.detach(), 'value_estimates': value_estimate.detach(), 'actions': torch.argmax(expert_probs, dim=1, keepdim=True).detach(), 'log_probs': torch.log(expert_probs + 1e-10).detach()}
                else:
                    policy_info = expert_probs.detach()
                return (combined_output, policy_info)
            else:
                if collect_policy_info:
                    dist = torch.distributions.Categorical(expert_probs)
                    actions = dist.sample()
                    one_hot = F.one_hot(actions, num_classes=7).float()
                    log_probs = dist.log_prob(actions).detach().unsqueeze(-1)
                    expert_weights = one_hot
                    policy_info = {'expert_probs': expert_probs.detach(), 'value_estimates': value_estimate.detach(), 'actions': actions.detach().unsqueeze(-1), 'log_probs': log_probs}
                else:
                    expert_weights = expert_probs
                    policy_info = expert_probs.detach()
                combined_output = torch.einsum('be,beh->bh', expert_weights, expert_outputs)
                with torch.no_grad():
                    self.expert_usage += expert_weights.sum(dim=0).detach()
                    self.expert_usage_history = 0.99 * self.expert_usage_history + 0.01 * expert_weights.sum(dim=0).detach()
                    self.total_steps += expert_weights.size(0)
                return (combined_output, policy_info)
        except Exception as e:
            raise

    def evaluate_actions(self, img, text, time, actions):
        """
        Evaluate actions for PPO - re-runs forward computation with detached inputs
        to get fresh action probabilities and value estimates
        """
        with torch.set_grad_enabled(True):
            img_feat = self.img_projector(img)
            text_feat = self.text_projector(text)
            time_feat = self.time_projector(time)
            gate_input = torch.cat([img_feat, text_feat, time_feat], dim=1)
            gate_logits = self.ppo_gate(gate_input)
            expert_probs = F.softmax(gate_logits / 1.0, dim=1)
            value_estimate = self.value_network(gate_input)
            diversity_bonus = self.get_diversity_bonus(expert_probs).unsqueeze(1)
            if isinstance(actions, torch.Tensor) and actions.dim() > 0:
                if actions.dim() > 1 and actions.size(1) > 1:
                    actions = actions[:, 0]
                actions = actions.long().reshape(-1)
                batch_indices = torch.arange(expert_probs.size(0), device=expert_probs.device)
                action_probs = expert_probs[batch_indices, actions].unsqueeze(-1)
            else:
                action_probs = expert_probs
            return (action_probs, value_estimate, expert_probs)

    def set_gate_temperature(self, temperature):
        """Set temperature for gate softmax"""
        self.gate_temperature = temperature

    def get_gate_temperature(self):
        """Get current gate temperature"""
        return self.gate_temperature
