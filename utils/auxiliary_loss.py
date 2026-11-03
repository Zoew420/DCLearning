import torch.nn.functional as F
import torch

def compute_kl_distillation_loss(student_features, teacher_features, temperature=2.0):
    """
    计算student和teacher特征之间的KL散度损失
    
    Args:
        student_features: 学生模型特征 (large_norm)
        teacher_features: 教师模型特征 (small_norm)
        temperature: 温度参数，用于软化概率分布
    
    Returns:
        kl_loss: KL散度损失值
    """
    # 将特征通过softmax转换为概率分布
    # 首先除以温度参数使分布更软
    soft_student = F.log_softmax(student_features / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_features / temperature, dim=-1)
    
    # 计算KL散度 (PyTorch的KL散度实现要求log_softmax和softmax)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return kl_loss

def convert_loss_to_reward(loss, method='negative', scale=1.0, baseline=None, min_reward=0.0):
    """
    将损失转换为奖励，使得损失越小，奖励越高
    
    Args:
        loss: 计算得到的KL散度损失
        method: 转换方法，可选 'negative'(取负), 'inverse'(取倒数), 'exp'(指数变换)
        scale: 奖励的缩放因子
        baseline: 基线值，如果提供，奖励将相对于基线计算
        min_reward: 最小奖励值，确保奖励不会低于此值
    
    Returns:
        reward: 转换后的奖励值
    """
    if method == 'negative':
        # 简单取负，损失越小，奖励越高
        reward = -loss * scale
    
    elif method == 'inverse':
        # 取倒数，损失接近0时奖励很大，需要加一个小值避免除零
        epsilon = 1e-6
        reward = scale / (loss + epsilon)
    
    elif method == 'exp':
        # 指数变换，损失越小，奖励越高
        reward = scale * torch.exp(-loss)
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # 如果提供了基线，对奖励进行基线化处理
    if baseline is not None:
        reward = reward - baseline
    
    # 确保奖励不低于最小值
    if min_reward is not None:
        reward = torch.max(reward, reward.new_tensor(min_reward))
    
    return reward
