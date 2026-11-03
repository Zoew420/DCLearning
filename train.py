import argparse
import json
from pathlib import Path

from common import build_model, get_device, load_dataset, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train MIMIC-IV-MM PPO using train/validation splits only.')
    parser.add_argument('--task', choices=['mortality', 'longstay', 'readmission'], default='readmission')
    parser.add_argument('--small_model_path', required=True, help='Matching task-specific small-model state_dict containing cls_head.*')
    parser.add_argument('--large_data_path', default='./data/large_embed', help='Root of large-model features and labels')
    parser.add_argument('--small_data_path', default='./data/MultimodalPred_input', help='Root of small-model features and labels')
    parser.add_argument('--save_path', default='./outputs/mimic_iv_mm')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--ppo_learning_rate', type=float, default=3e-4)
    parser.add_argument('--ppo_updates_per_epoch', type=int, default=4)
    parser.add_argument('--distillation_weight', type=float, default=0.2)
    parser.add_argument('--reward_type', choices=['negative', 'inverse', 'exp'], default='exp')
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--cls_dropout', type=float, default=0.1)
    parser.add_argument('--gate_temperature', type=float, default=0.5, help='Inference temperature; training uses at least 1.0')
    parser.add_argument('--large_image_model_name', choices=['dinov2', 'rad-dino'], default='dinov2')
    parser.add_argument('--large_text_model_name', choices=['MeLLaMA-13B', 'BioMistral-7B'], default='MeLLaMA-13B')
    parser.add_argument('--large_time_model_name', default='moirai-1.1-R-large')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.ppo_updates_per_epoch) < 1:
        parser.error('epochs, batch_size and ppo_updates_per_epoch must be positive')
    if not 0 < args.data_ratio <= 1:
        parser.error('data_ratio must be in (0, 1]')
    if args.hidden_dim < 8 or (3 * args.hidden_dim) % 8:
        parser.error('3 * hidden_dim must be divisible by 8')
    if args.gate_temperature <= 0 or not 0 <= args.distillation_weight <= 1:
        parser.error('gate_temperature must be positive and distillation_weight in [0, 1]')
    return args


def main(args):
    global torch, nn, F, np, Subset, autocast, logger, use_amp, compute_ppo_reward
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from torch.cuda.amp import autocast
    from models.SimplifiedPPO_MoE_Agent import compute_ppo_reward
    from utils.logger import get_logger

    device = get_device(args.device)
    set_seed(args.seed)
    logger = get_logger('train', 'MIMIC-IV-MM')
    use_amp = False  # Original training uses float32; use it consistently for validation.
    output = Path(args.save_path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best_model.pth').exists():
        raise FileExistsError('Choose a new --save_path to preserve the previous checkpoint.')
    config = vars(args).copy()
    config.update(image_dim={'dinov2': 1536, 'rad-dino': 768}[args.large_image_model_name],
                  text_dim={'MeLLaMA-13B': 5120, 'BioMistral-7B': 4960}[args.large_text_model_name], time_dim=1024)
    train_data = load_dataset('train', config)
    val_data = load_dataset('val', config)
    if args.data_ratio < 1:
        size = max(1, int(len(train_data) * args.data_ratio))
        indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed))[:size].sort().values.tolist()
        train_data = Subset(train_data, indices)
    labels = np.asarray([train_data.dataset.label[i] for i in train_data.indices]
                        if isinstance(train_data, Subset) else train_data.label)
    val_labels = np.asarray(val_data.label)
    if not len(labels) or set(np.unique(val_labels)) != {0, 1}:
        raise ValueError('Training must be non-empty and validation must contain both classes.')
    positive_ratio = float((labels.sum() + val_labels.sum()) / (len(labels) + len(val_labels)))
    sampler = MultiModalImbalancedDatasetSampler(
        train_data, generator=torch.Generator().manual_seed(args.seed + 1))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    model = build_model(config, device, args.small_model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ppo_parameters = list(model.large_moe_model.ppo_gate.parameters()) + list(model.large_moe_model.value_network.parameters())
    ppo_optimizer = torch.optim.Adam(ppo_parameters, lr=args.ppo_learning_rate)
    # Effective values used by the original main_ppo.py.
    model.set_ppo_optimizer(optimizer=ppo_optimizer, batch_size=args.batch_size,
                            ppo_epochs=5, clip_ratio=0.1, gamma=0.95, lam=0.95,
                            entropy_coef=0.02, value_coef=0.75, max_grad_norm=0.5)
    class_weights = {'mortality': [1, 2], 'longstay': [1, 1], 'readmission': [1, 5]}
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights[args.task], dtype=torch.float32, device=device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    save_json(output / 'config.json', config)
    history = []
    best_auroc = -float('inf')
    for epoch in range(args.epochs):
        loss, ppo_metrics = train_ppo_epoch(epoch, model, device, train_loader, criterion, optimizer, None, args)
        scheduler.step()
        auroc = float(val_epoch(epoch, model, device, val_loader, args))
        if not np.isfinite(loss) or not np.isfinite(auroc):
            raise FloatingPointError('Non-finite training loss or validation AUROC.')
        history.append({'epoch': epoch + 1, 'train_loss': float(loss), 'val_auroc': auroc, 'ppo': ppo_metrics})
        save_json(output / 'history.json', history)
        logger.info('Epoch %d: loss=%.5f validation AUROC=%.5f', epoch + 1, loss, auroc)
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({'model_state_dict': model.state_dict(), 'config': config,
                        'epoch': epoch + 1, 'best_val_auroc': best_auroc,
                        'positive_ratio': positive_ratio}, output / 'best_model.pth')
    logger.info('Best validation AUROC %.5f. Checkpoint: %s', best_auroc, output / 'best_model.pth')


if __name__ == '__main__':
    args = parse_args()
# Import after CLI parsing so --help works without the research dependencies.
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset


class MultiModalImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, generator=None):
        """
        多模态数据的过采样采样器
        Args:
            dataset: NPY_Large_Small_Mul_Dataset实例
            indices: 可选的索引列表，用于指定要使用的样本
            generator: PyTorch随机数生成器
        """
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices)
        self.generator = generator
        if isinstance(dataset, Subset):
            self.labels = [dataset.dataset.label[i] for i in dataset.indices]
        else:
            self.labels = dataset.label.tolist()
        label_to_count = {}
        for idx in self.indices:
            label = self.labels[idx]
            label_to_count[label] = label_to_count.get(label, 0) + 1
        weights = []
        for idx in self.indices:
            label = self.labels[idx]
            weights.append(1.0 / label_to_count[label])
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True, generator=self.generator))

    def __len__(self):
        return self.num_samples

def train_ppo_epoch(epoch, model, device, train_loader, task_criterion, optimizer, scaler, args):
    model.train()
    total_task_loss = 0
    total_ppo_metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0, 'reward_mean': 0, 'reward_std': 0}
    num_samples = 0
    nan_batches = 0
    ppo_buffer_full = False
    batches_per_ppo_update = max(1, len(train_loader) // args.ppo_updates_per_epoch)
    for batch_idx, (large_img, large_text, large_time, small_feat, label) in enumerate(train_loader):
        try:
            torch.cuda.empty_cache()
            buffer_size = model.ppo_trainer.buffer.size() if model.ppo_trainer else 0
            if ppo_buffer_full and buffer_size == 0:
                logger.info('PPO update just completed, resetting computation graph state')
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                ppo_buffer_full = False
            large_img = large_img.to(device, dtype=torch.float32)
            large_text = large_text.to(device, dtype=torch.float32)
            large_time = large_time.to(device, dtype=torch.float32)
            small_feat = small_feat.to(device, dtype=torch.float32)
            label = label.to(device)
            with autocast(enabled=False):
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    large_logits, large_feat, policy_outputs = model(large_img, large_text, large_time, small_feat, training=True)
                    if torch.isnan(large_logits).any() or torch.isinf(large_logits).any():
                        nan_batches += 1
                        logger.warning(f'Warning: NaN/Inf detected in large_logits for batch {batch_idx}')
                        continue
                    task_loss = task_criterion(large_logits, label)
                    total_task_loss += task_loss.item() * label.size(0)
                    task_loss.backward()
                    optimizer.step()
            with torch.no_grad():
                _, predicted = torch.max(large_logits, 1)
                accuracy = (predicted == label).float().mean().item()
                performance_reward = accuracy
                feature_reward = compute_ppo_reward(args, large_feat, small_feat, task_loss.item(), distillation_weight=args.distillation_weight)
                performance_weight = 0.3
                reward = (1 - performance_weight) * feature_reward + performance_weight * performance_reward
                if isinstance(reward, torch.Tensor) and reward.numel() > 1:
                    reward = reward.mean().item()
                elif isinstance(reward, torch.Tensor):
                    reward = reward.item()
                states = {'img': large_img.detach().clone(), 'text': large_text.detach().clone(), 'time': large_time.detach().clone(), 'small_feat': small_feat.detach().clone()}
                value_key = 'value_estimates' if 'value_estimates' in policy_outputs else 'value_estimate'
                log_probs_key = 'log_probs' if 'log_probs' in policy_outputs else None
                if log_probs_key is None:
                    logger.warning('No log_probs found in policy_outputs, creating default ones')
                    uniform_log_probs = torch.log(torch.ones(large_img.size(0), 1) / 7.0).to(device, dtype=torch.float32)
                    policy_outputs['log_probs'] = uniform_log_probs
                    log_probs_key = 'log_probs'
                if isinstance(policy_outputs['actions'], torch.Tensor):
                    actions = policy_outputs['actions'].detach().clone().to(dtype=torch.float32)
                else:
                    actions = policy_outputs['actions']
                if isinstance(policy_outputs[value_key], torch.Tensor):
                    values = policy_outputs[value_key].detach().clone().to(dtype=torch.float32)
                else:
                    values = policy_outputs[value_key]
                if isinstance(policy_outputs[log_probs_key], torch.Tensor):
                    log_probs = policy_outputs[log_probs_key].detach().clone().to(dtype=torch.float32)
                else:
                    log_probs = policy_outputs[log_probs_key]
                model.store_transition(states, actions, reward, values, log_probs, False)
            buffer_size = model.ppo_trainer.buffer.size() if model.ppo_trainer else 0
            if buffer_size >= 2048 or ((batch_idx + 1) % batches_per_ppo_update == 0 and buffer_size > 0):
                ppo_buffer_full = True
            if ppo_buffer_full and batch_idx < len(train_loader) - 1:
                try:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    ppo_metrics = model.update_policy()
                    for k, v in ppo_metrics.items():
                        if k in total_ppo_metrics:
                            total_ppo_metrics[k] += v
                    torch.cuda.empty_cache()
                    logger.info(f'Policy update completed with metrics: {ppo_metrics}')
                except Exception as e:
                    raise
            num_samples += large_img.size(0)
        except RuntimeError as e:
            raise
        except Exception as e:
            raise
    if model.ppo_trainer and model.ppo_trainer.buffer.size() > 0:
        try:
            ppo_metrics = model.update_policy()
            for k, v in ppo_metrics.items():
                if k in total_ppo_metrics:
                    total_ppo_metrics[k] += v
            logger.info(f'Final policy update completed with metrics: {ppo_metrics}')
        except Exception as e:
            raise
    if nan_batches > 0:
        logger.warning(f'Skipped {nan_batches} batches due to errors or NaN/Inf values.')
    avg_task_loss = total_task_loss / (num_samples or 1)
    avg_ppo_metrics = {k: v / (args.ppo_updates_per_epoch or 1) for k, v in total_ppo_metrics.items()}
    return (avg_task_loss, avg_ppo_metrics)

@torch.no_grad()
def val_epoch(epoch, model, device, val_loader, args):
    model.eval()
    all_targets = []
    all_preds = []
    all_expert_weights = []
    for batch_idx, (large_image, large_text, large_timeseries, small_feat, target) in enumerate(val_loader):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            large_image = large_image.to(device)
            large_text = large_text.to(device)
            large_timeseries = large_timeseries.to(device)
            small_feat = small_feat.to(device)
            target = target.to(device)
            large_logits, large_feat, expert_weights = model(large_image, large_text, large_timeseries, small_feat, training=False)
            all_targets.append(target)
            all_preds.append(large_logits.to('cpu'))
            if isinstance(expert_weights, torch.Tensor):
                all_expert_weights.append(expert_weights.detach().to('cpu'))
            elif isinstance(expert_weights, dict) and 'expert_probs' in expert_weights:
                all_expert_weights.append(expert_weights['expert_probs'].detach().to('cpu'))
            else:
                uniform_weights = torch.ones(large_image.size(0), 7).to('cpu') / 7.0
                all_expert_weights.append(uniform_weights)
    all_targets = torch.cat(all_targets).to('cpu').float().numpy()
    all_preds = torch.cat(all_preds).float()
    all_probs = torch.softmax(all_preds, dim=1)[:, 1].to('cpu').numpy()
    all_expert_weights = torch.cat(all_expert_weights)
    auroc = roc_auc_score(all_targets, all_probs)
    expert_names = ['Image Only', 'Text Only', 'Time Only', 'Image+Text', 'Image+Time', 'Text+Time', 'Image+Text+Time']
    avg_expert_weights = all_expert_weights.mean(dim=0).cpu().numpy()
    weight_str = ', '.join([f'{name}: {weight:.4f}' for name, weight in zip(expert_names, avg_expert_weights)])
    logger.info(f'Validation expert weights: {weight_str}')
    return auroc

if __name__ == '__main__':
    main(args)
