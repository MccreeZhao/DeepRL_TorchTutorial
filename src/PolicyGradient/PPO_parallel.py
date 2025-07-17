import math
import random
from collections import deque
from collections import namedtuple
from itertools import count
import os

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import time
import numpy as np


# 1. Initialize the environment
# env = gym.make("CartPole-v1")
def make_env(seed):
    def _thunk():
        e = gym.make("CartPole-v1")
        e.reset(seed=seed)
        return e
    return _thunk

num_envs = 8
env = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, obs, act, logp, rew, done, value):
        self.observations.append(obs)
        self.actions.append(act)
        self.logprobs.append(logp)
        self.rewards.append(rew)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.observations, self.actions, self.logprobs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def compute_returns_advantages(self, next_value, gamma=0.99, lam=0.95):
        """GAE‑λ; returns tensor advantages normalized, returns."""
        values = self.values + [next_value]
        advantages = []
        gae = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step+1] * (1 - self.dones[step].float()) - values[step]
            gae = delta + gamma * lam * (1 - self.dones[step].float()) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, self.values)]
        # to tensors
        self.observations = torch.cat(self.observations, dim=0).to(device)
        self.actions = torch.cat(self.actions, dim=0).to(device)
        self.logprobs = torch.cat(self.logprobs, dim=0).detach()
        self.advantages = torch.cat(advantages, dim=0).flatten().to(dtype=torch.float32, device=device)
        self.returns = torch.cat(returns, dim=0).flatten().to(dtype=torch.float32, device=device)
        self.values = torch.cat(self.values, dim=0).flatten().to(dtype=torch.float32, device=device)

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-8)



class Actor(nn.Module):
    """Actor网络，用于生成动作概率分布"""
    def __init__(self, num_obs: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        # self.softmax = nn.Softmax(dim=1)
        
        # 更好的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """Critic网络，用于评估状态价值"""
    def __init__(self, num_obs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 更好的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# 4. Training Hyperparameters and Helper Functions
# 外循环 outer loop
TOTAL_TIMESTEPS = 300_000
# Outer Epoch = TOTAL_TIMESTEPS / ROLLOUT_STEPS
# 内循环 inner loop
ROLLOUT_STEPS = 2048
MINI_EPOCH = 16 #对MINI_EPOCH和MINIBATCH_SIZE还挺敏感的，如果batch_size太大，收敛很慢，batch_size太小，可能会震荡
MINIBATCH_SIZE = 256

# PPO:
GAMMA = 0.99
LAMBDA = 0.95
CLIP_COEF = 0.2  # Policy clipping coefficient
VALUE_CLIP_COEF = 0.2  # Value clipping coefficient (可以调整为不同值)
VALUE_COEF = 0.5
ENTROPY_COEF_INITIAL = 0.5  # 初始entropy系数
ENTROPY_COEF_FINAL = 0.2   # 最终entropy系数（衰减到0）
# ENTROPY_COEF会在训练过程中动态计算


critic_lr = 1e-2 #如果开value clip, 可以调成5e-2
actor_lr = 1e-2
grad_norm_clip = 1

# 计算总的训练epoch数
TOTAL_EPOCHS = TOTAL_TIMESTEPS // ROLLOUT_STEPS

def get_entropy_coef(current_epoch, total_epochs=TOTAL_EPOCHS, 
                     initial_coef=ENTROPY_COEF_INITIAL, final_coef=ENTROPY_COEF_FINAL):
    """
    计算当前epoch的entropy coefficient
    支持线性衰减、指数衰减和余弦衰减三种模式
    """
    progress = current_epoch / total_epochs
    progress = min(progress, 1.0)  # 确保不超过1.0
    
    # 线性衰减 (推荐)
    entropy_coef = initial_coef + (final_coef - initial_coef) * progress
    
    # 可选的其他衰减方式：
    # 指数衰减: entropy_coef = initial_coef * (0.01 ** progress)
    # 余弦衰减: entropy_coef = final_coef + (initial_coef - final_coef) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return entropy_coef



# Get number of actions from gym action space
n_actions = env.single_action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = env.single_observation_space.shape[0]

# actor = Actor(n_observations, n_actions).to(device)
# critic = Critic(n_observations).to(device)
actor = Actor(n_observations, n_actions).to(device)
critic = Critic(n_observations).to(device)
# optimizer = optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=LR, amsgrad=True)
critic_optimizer = optim.AdamW(critic.parameters(), lr=critic_lr, amsgrad=True)
actor_optimizer = optim.AdamW(actor.parameters(), lr=actor_lr, amsgrad=True)

# memory = ReplayMemory(1000)
buffer = RolloutBuffer()
# memory.clear()


def select_action(state):
    logits = actor(state)
    value = critic(state).detach()
    dist = torch.distributions.Categorical(logits=logits)
    entropy = dist.entropy()
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob, entropy, value


def test_model(num_test_episodes=3):
    """测试模型性能，使用确定性动作"""
    actor.eval()  # 设置为评估模式
    test_durations_local = []
    
    test_env = gym.make("CartPole-v1")
    
    for _ in range(num_test_episodes):
        state, info = test_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            with torch.no_grad():
                # 使用确定性动作：选择概率最高的动作
                prob = actor(state)
                action = prob.argmax(dim=1).item()
            
            observation, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            
            if done:
                test_durations_local.append(t + 1)
                break
                
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
    test_env.close()
    actor.train()  # 恢复训练模式
    return sum(test_durations_local) / len(test_durations_local)  # 返回平均持续时间


episode_durations = []
# 全局记录每个 episode 的平均 loss
episode_losses = []
# 添加记录三个分别的loss
episode_policy_losses = []
episode_value_losses = []
episode_entropy_losses = []
# 添加记录entropy coefficient的变化
episode_entropy_coefs = []
# 添加测试相关的记录
test_episodes = []  # 记录测试的episode编号
test_durations = []  # 记录测试的持续时间
test_interval = 10  # 每10轮测试一次
early_stop_threshold = 500  # 提前停止阈值
full_trajs_times = 0
early_stopped = False  # 提前停止标志


def plot_durations(show_result=False):
    plt.figure(1, figsize=(12, 5))
    plt.clf()  # Clear the figure at the start to prevent duplicate elements
    
    # 绘制训练曲线
    plt.subplot(1, 2, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Training Result")
    else:
        plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy(), label='Training', alpha=0.7)
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='Training (100-ep avg)', linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制测试曲线
    plt.subplot(1, 2, 2)
    if len(test_durations) > 0:
        test_durations_t = torch.tensor(test_durations, dtype=torch.float)
        test_episodes_t = torch.tensor(test_episodes, dtype=torch.float)
        if show_result:
            plt.title("Test Result")
        else:
            plt.title("Test Progress")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(test_episodes_t.numpy(), test_durations_t.numpy(), 'ro-', label='Test', linewidth=2, markersize=6)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        if show_result:
            plt.title("Test Result (No data)")
        else:
            plt.title("Test Progress (No data yet)")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_losses(show_result=False):
    """绘制三个loss曲线和entropy coefficient在同一幅图的四个子图中"""
    plt.figure(2, figsize=(20, 5))
    plt.clf()  # Clear the figure at the start to prevent duplicate elements
    
    if show_result:
        plt.suptitle("Training Losses Result")
    else:
        plt.suptitle("Training Losses Progress")
    
    # 策略损失
    plt.subplot(1, 4, 1)
    if len(episode_policy_losses) > 0:
        policy_losses_t = torch.tensor(episode_policy_losses, dtype=torch.float)
        plt.title("Policy Loss")
        plt.xlabel("Episode")
        plt.ylabel("Policy Loss")
        plt.plot(policy_losses_t.numpy(), label='Policy Loss', color='red', alpha=0.7)
        # 计算移动平均
        if len(policy_losses_t) >= 10:
            means = policy_losses_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy(), label='Policy Loss (10-ep avg)', color='darkred', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title("Policy Loss (No data yet)")
        plt.xlabel("Episode")
        plt.ylabel("Policy Loss")
        plt.grid(True, alpha=0.3)
    
    # 价值损失
    plt.subplot(1, 4, 2)
    if len(episode_value_losses) > 0:
        value_losses_t = torch.tensor(episode_value_losses, dtype=torch.float)
        plt.title("Value Loss")
        plt.xlabel("Episode")
        plt.ylabel("Value Loss")
        plt.plot(value_losses_t.numpy(), label='Value Loss', color='blue', alpha=0.7)
        # 计算移动平均
        if len(value_losses_t) >= 10:
            means = value_losses_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy(), label='Value Loss (10-ep avg)', color='darkblue', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title("Value Loss (No data yet)")
        plt.xlabel("Episode")
        plt.ylabel("Value Loss")
        plt.grid(True, alpha=0.3)
    
    # 熵损失
    plt.subplot(1, 4, 3)
    if len(episode_entropy_losses) > 0:
        entropy_losses_t = torch.tensor(episode_entropy_losses, dtype=torch.float)
        plt.title("Entropy Loss")
        plt.xlabel("Episode")
        plt.ylabel("Entropy Loss")
        plt.plot(entropy_losses_t.numpy(), label='Entropy Loss', color='green', alpha=0.7)
        # 计算移动平均
        if len(entropy_losses_t) >= 10:
            means = entropy_losses_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy(), label='Entropy Loss (10-ep avg)', color='darkgreen', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title("Entropy Loss (No data yet)")
        plt.xlabel("Episode")
        plt.ylabel("Entropy Loss")
        plt.grid(True, alpha=0.3)
    
    # Entropy Coefficient衰减曲线
    plt.subplot(1, 4, 4)
    if len(episode_entropy_coefs) > 0:
        entropy_coefs_t = torch.tensor(episode_entropy_coefs, dtype=torch.float)
        plt.title("Entropy Coefficient Decay")
        plt.xlabel("Episode")
        plt.ylabel("Entropy Coefficient")
        plt.plot(entropy_coefs_t.numpy(), label='Entropy Coef', color='purple', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 添加理论衰减曲线作为参考
        episodes = list(range(len(episode_entropy_coefs)))
        theoretical_coefs = [get_entropy_coef(ep) for ep in episodes]
        plt.plot(episodes, theoretical_coefs, '--', label='Theoretical', color='orange', alpha=0.7)
        plt.legend()
    else:
        plt.title("Entropy Coefficient (No data yet)")
        plt.xlabel("Episode")
        plt.ylabel("Entropy Coefficient")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# 5. Training Function
def optimize_model(i_episode, buffer, entropy_coef):

    b_states = buffer.observations
    b_actions = buffer.actions
    b_logprobs = buffer.logprobs
    b_rewards = buffer.rewards
    b_dones = buffer.dones
    b_values = buffer.values
    b_advantages = buffer.advantages
    b_returns = buffer.returns

    # 计算 value loss
    data_length = len(b_states)
    indexes = torch.arange(data_length)
    
    # 用于累积loss值
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    total_loss_accumulated = 0
    num_updates = 0
    
    for epoch in range(MINI_EPOCH):
        # np.random.shuffle(indexes)
        indexes = torch.randperm(data_length, device=device)
        for batch_start in range(0, data_length, MINIBATCH_SIZE):
            end = batch_start + MINIBATCH_SIZE
            batch_indexes = indexes[batch_start:end]
            b_states_batch = b_states[batch_indexes]
            b_actions_batch = b_actions[batch_indexes]
            b_logprobs_batch = b_logprobs[batch_indexes]
            b_advantages_batch = b_advantages[batch_indexes]
            b_returns_batch = b_returns[batch_indexes]
            b_values_batch = b_values[batch_indexes]  # 旧的value值

            logits = actor(b_states_batch)
            values = critic(b_states_batch).squeeze()  # 新的value值
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy()
            log_probs = dist.log_prob(b_actions_batch)

            ratio = (log_probs - b_logprobs_batch).exp()
            surr1 = ratio * b_advantages_batch
            surr2 = torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * b_advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss with clipping (PPO style)
            # # 对value function也进行clipping，防止更新过于激进
            # clipped_values = b_values_batch + torch.clamp(values - b_values_batch, -VALUE_CLIP_COEF, VALUE_CLIP_COEF)
            # value_loss_unclipped = (values-b_returns_batch).pow(2) #F.mse_loss(values, b_returns_batch)
            # value_loss_clipped = (clipped_values-b_returns_batch).pow(2) #F.mse_loss(clipped_values, b_returns_batch)
            # # value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
            # value_loss = torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
            # # 如果更新太快，unclipped loss就会比较小，这时候我们应该用clipped loss，来选择更保守的更新
            # # Value loss without clipping
            value_loss = F.mse_loss(values, b_returns_batch)

            entropy_loss = dist.entropy().mean()

            total_loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy_loss
            
            # 累积loss值
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss_accumulated += total_loss.item()
            num_updates += 1
            
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_norm_clip)
            critic_optimizer.step()
            actor_optimizer.step()

    # 计算平均loss
    avg_policy_loss = total_policy_loss / num_updates
    avg_value_loss = total_value_loss / num_updates
    avg_entropy_loss = total_entropy_loss / num_updates
    avg_total_loss = total_loss_accumulated / num_updates

    print(f"i_episode: {i_episode}, policy_loss: {avg_policy_loss:.4f}, value_loss: {avg_value_loss:.4f}, entropy_loss: {avg_entropy_loss:.4f}, entropy_coef: {entropy_coef:.5f}, total_loss: {avg_total_loss:.4f}")

    return avg_total_loss, avg_policy_loss, avg_value_loss, avg_entropy_loss


# 创建保存文件夹
save_dir = "save/PPO_parallel"
os.makedirs(save_dir, exist_ok=True)

# 打印训练配置信息
print(f"Training Configuration:")
print(f"Total Timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Rollout Steps: {ROLLOUT_STEPS:,}")
print(f"Total Epochs: {TOTAL_EPOCHS}")
print(f"Entropy Coefficient Decay: {ENTROPY_COEF_INITIAL} → {ENTROPY_COEF_FINAL}")
print(f"Initial Entropy Coef: {get_entropy_coef(0):.5f}")
print(f"Mid-training Entropy Coef: {get_entropy_coef(TOTAL_EPOCHS//2):.5f}")
print(f"Final Entropy Coef: {get_entropy_coef(TOTAL_EPOCHS-1):.5f}")
print(f"Early Stopping:")
print(f"  Enabled: Yes")
print(f"  Threshold: {early_stop_threshold} (test duration)")
print(f"  Test Interval: Every {test_interval} episodes")
print("-" * 60)

global_step = 0

for i_episode in tqdm(range(TOTAL_TIMESTEPS//ROLLOUT_STEPS)):
    episode_returns, episode_lengths = [], []
    episode_return = np.zeros(num_envs)
    episode_length = np.zeros(num_envs)
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    episode_loss_values = []  # 记录当前 episode 内所有优化步骤的loss
    buffer.clear()

    for rollout_step in range(ROLLOUT_STEPS // num_envs):
        global_step += num_envs
        action, log_prob, entropy, value = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        
        # Reward Shaping
        position = np.abs(observation[:, 0])
        velocity = np.abs(observation[:, 1])
        penalty = np.where(position < 0.1, 0, (position - 0.1) * 0.2 + velocity * 0.02)
        reward -= penalty
        
        done = np.logical_or(terminated, truncated)

        buffer.store(state, action, log_prob, torch.tensor(reward, dtype=torch.float32, device=device), torch.tensor(done, dtype=torch.bool, device=device), value.squeeze(-1))

        next_state = torch.tensor(observation, dtype=torch.float32, device=device)

        # Handle resets
        reset_ids = np.where(done)[0]
        for idx in reset_ids:
            obs, _ = env.envs[idx].reset()
            next_state[idx] = torch.tensor(obs, dtype=torch.float32, device=device)

        state = next_state

        episode_return += reward
        episode_length += 1

        if len(reset_ids) > 0:
            for idx in reset_ids:
                episode_returns.append(episode_return[idx])
                episode_lengths.append(episode_length[idx])
                print(f"Env {idx} episode_length: {episode_length[idx]}")
                episode_durations.append(episode_length[idx])
                episode_return[idx] = 0
                episode_length[idx] = 0
    with torch.no_grad():
        next_value = critic(state).squeeze(-1).detach()
    buffer.compute_returns_advantages(next_value, GAMMA, LAMBDA)

    # 计算当前epoch的entropy coefficient
    current_entropy_coef = get_entropy_coef(i_episode)
    
    total_loss_val, policy_loss_val, value_loss_val, entropy_loss_val = optimize_model(i_episode, buffer, current_entropy_coef)
    episode_loss_values.append(total_loss_val)
    
    # 记录三个分别的loss
    episode_policy_losses.append(policy_loss_val)
    episode_value_losses.append(value_loss_val)
    episode_entropy_losses.append(entropy_loss_val)
    # 记录当前的entropy coefficient
    episode_entropy_coefs.append(current_entropy_coef)
    
    if episode_loss_values:
        avg_loss = sum(episode_loss_values) / len(episode_loss_values)
    else:
        avg_loss = None
    episode_losses.append(avg_loss)
    print(f"Episode {i_episode + 1}: Duration = {np.mean(episode_lengths):.2f}, Average Loss = {avg_loss:.4f}, Entropy Coef = {current_entropy_coef:.5f}")

    # 每20轮执行一次测试
    if (i_episode + 1) % test_interval == 0:
        avg_test_duration = test_model(num_test_episodes=20)
        test_episodes.append(i_episode + 1)
        test_durations.append(avg_test_duration)
        print(f"Test at Episode {i_episode + 1}: Average Duration = {avg_test_duration:.2f}")
        
        # 检查是否达到提前停止条件
        if avg_test_duration >= early_stop_threshold:
            full_trajs_times += 1
            if full_trajs_times >= 1:
                print(f"🎉 Early stopping triggered! Test duration {avg_test_duration:.2f} >= {early_stop_threshold}")
                print(f"Training completed at episode {i_episode + 1}/{TOTAL_EPOCHS}")
                early_stopped = True
                break
    
    plot_durations()
    plot_losses()

print("Complete")
print(f"Training Summary:")
if early_stopped:
    print(f"✅ Early stopping activated - Target performance reached!")
    print(f"   Stopped at episode: {len(episode_durations)}")
    print(f"   Final test duration: {test_durations[-1]:.2f}")
    print(f"   Target threshold: {early_stop_threshold}")
else:
    print(f"📊 Regular training completion")
    print(f"   Total episodes: {len(episode_durations)}")
    if test_durations:
        print(f"   Final test duration: {test_durations[-1]:.2f}")

print(f"Entropy Decay Summary:")
print(f"Started with entropy coef: {episode_entropy_coefs[0]:.5f}")
print(f"Ended with entropy coef: {episode_entropy_coefs[-1]:.5f}")
print(f"Total decay: {episode_entropy_coefs[0] - episode_entropy_coefs[-1]:.5f}")
print("-" * 60)
# plot_durations(show_result=True)
plot_durations(show_result=True)
plot_losses(show_result=True)
plt.ioff()
plt.show()

# 保存训练和测试曲线图
plt.figure(1)
filename_suffix = "_early_stopped" if early_stopped else "_full_training"
plt.savefig(os.path.join(save_dir, f"PPO_training_and_test{filename_suffix}.png"), dpi=300, bbox_inches='tight')

# 保存loss曲线图
plt.figure(2)
plt.savefig(os.path.join(save_dir, f"PPO_losses{filename_suffix}.png"), dpi=300, bbox_inches='tight')

# 执行最终测试
print("\nFinal Test:")
final_test_duration = test_model(num_test_episodes=10)
print(f"Final Test Result: Average Duration = {final_test_duration:.2f}")

# 保存测试结果和loss数据到文件
test_results = {
    'test_episodes': test_episodes,
    'test_durations': test_durations,
    'final_test_duration': final_test_duration,
    'episode_policy_losses': episode_policy_losses,
    'episode_value_losses': episode_value_losses,
    'episode_entropy_losses': episode_entropy_losses,
    'episode_total_losses': episode_losses,
    'episode_entropy_coefs': episode_entropy_coefs,
    'entropy_decay_config': {
        'initial_coef': ENTROPY_COEF_INITIAL,
        'final_coef': ENTROPY_COEF_FINAL,
        'total_epochs': TOTAL_EPOCHS
    },
    'early_stopping_info': {
        'early_stopped': early_stopped,
        'threshold': early_stop_threshold,
        'actual_episodes': len(episode_durations),
        'planned_episodes': TOTAL_EPOCHS,
        'final_test_performance': test_durations[-1] if test_durations else None
    }
}
import json
json_filename = f'test_results{filename_suffix}.json'
with open(os.path.join(save_dir, json_filename), 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"Test results saved to {os.path.join(save_dir, json_filename)}")

# 保存训练好的模型
actor_filename = f"Actor_cartpole{filename_suffix}.pth"
critic_filename = f"Critic_cartpole{filename_suffix}.pth"
actor_path = os.path.join(save_dir, actor_filename)
critic_path = os.path.join(save_dir, critic_filename)
torch.save(actor.state_dict(), actor_path)
torch.save(critic.state_dict(), critic_path)
print(f"Models saved as {actor_path} and {critic_path}")

# ----------------- 模型加载并生成视频演示 -----------------
# 创建新环境用于录制视频，这里使用 gymnasium 提供的 RecordVideo 封装器

# 注意：创建环境时需要指定 render_mode 为 "rgb_array" 以便录制视频帧
video_env = gym.make("CartPole-v1", render_mode="rgb_array")
# RecordVideo 会自动保存视频到指定目录
current_time = time.strftime("%Y%m%d_%H%M%S")
video_name = f"cartpole_PPO_parallel_{current_time}{filename_suffix}"
video_env = RecordVideo(
    video_env, 
    video_folder=save_dir,
    name_prefix=video_name,
    episode_trigger=lambda episode_id: True
)

# 加载保存的模型
actor.load_state_dict(torch.load(actor_path, map_location=device))
critic.load_state_dict(torch.load(critic_path, map_location=device))
actor.eval()
critic.eval()

# 初始化环境并开始录制视频
state, info = video_env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False

while not done:
    # 根据当前状态选择动作（贪婪策略）
    with torch.no_grad():
        logits = actor(state)
        action = logits.argmax(dim=-1).item()
    observation, reward, terminated, truncated, _ = video_env.step(action)
    done = terminated or truncated
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

video_env.close()
print("Video saved in 'videos' folder")
