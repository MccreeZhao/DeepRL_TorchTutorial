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


# 1. Initialize the environment
env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 2. Establish the Replay Memory
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "log_prob", "entropy", "done"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def popleft(self):
        return self.memory.popleft()

    def popright(self):
        return self.memory.pop()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()


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
        self.softmax = nn.Softmax(dim=1)
        
        # 更好的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.softmax(self.net(x))


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
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor of the reward
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 1000
# TAU = 0.005
# LR = 3e-4
critic_lr = 1e-2
actor_lr = 2e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# actor = Actor(n_observations, n_actions).to(device)
# critic = Critic(n_observations).to(device)
actor = Actor(n_observations, n_actions).to(device)
critic = Critic(n_observations).to(device)
# optimizer = optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=LR, amsgrad=True)
critic_optimizer = optim.AdamW(critic.parameters(), lr=critic_lr, amsgrad=True, weight_decay=1e-4)
actor_optimizer = optim.AdamW(actor.parameters(), lr=actor_lr, amsgrad=True, weight_decay=1e-4)

memory = ReplayMemory(1000)
memory.clear()
steps_done = 0


def select_action(state):
    global steps_done
    steps_done += 1
    prob = actor(state)
    dist = torch.distributions.Categorical(probs=prob)
    entropy = dist.entropy()
    action = dist.sample()
    log_prob = dist.log_prob(action)  # better than torch.log() ?
    return action.item(), log_prob, entropy  # [1]


def test_model(num_test_episodes=3):
    """测试模型性能，使用确定性动作"""
    actor.eval()  # 设置为评估模式
    test_durations_local = []
    
    for _ in range(num_test_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            with torch.no_grad():
                # 使用确定性动作：选择概率最高的动作
                prob = actor(state)
                action = prob.argmax(dim=1).item()
            
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                test_durations_local.append(t + 1)
                break
                
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
    actor.train()  # 恢复训练模式
    return sum(test_durations_local) / len(test_durations_local)  # 返回平均持续时间


episode_durations = []
# 全局记录每个 episode 的平均 loss
episode_losses = []
# 添加测试相关的记录
test_episodes = []  # 记录测试的episode编号
test_durations = []  # 记录测试的持续时间
test_interval = 20  # 每20轮测试一次


def plot_durations(show_result=False):
    plt.figure(1, figsize=(12, 5))
    
    # 绘制训练曲线
    plt.subplot(1, 2, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Training Result")
    else:
        plt.clf()
        plt.subplot(1, 2, 1)
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


# 5. Training Function
def optimize_model(i_episode):

    accumulated_reward = []
    log_probs = []
    advantages = []
    cur_state = []
    next_state = []
    rewards = []
    entropies = []
    dones = []  # 添加done标记列表
    reward_sum = 0
    # if len(memory) < BATCH_SIZE:
    #     return None
    while len(memory) > 0:
        item = memory.popright()
        reward_sum = reward_sum * GAMMA + item.reward
        accumulated_reward.insert(0, reward_sum)
        log_probs.insert(0, item.log_prob)
        cur_state.insert(0, item.state)
        next_state.insert(0, item.next_state)
        rewards.insert(0, item.reward)
        entropies.insert(0, item.entropy)
        dones.insert(0, item.done)  # 添加done标记
    # accumulated_reward = torch.tensor(accumulated_reward, device=device, dtype=torch.float)
    # accumulated_reward = (accumulated_reward - accumulated_reward.mean()) / (accumulated_reward.std() + 1e-5)

    # 计算 value loss
    
    cur_state = torch.stack(cur_state, dim=0).to(device).squeeze(1)
    next_state = torch.stack(next_state, dim=0).to(device).squeeze(1)
    done_mask = torch.tensor(dones, device=device, dtype=torch.bool)  # 使用专门的done mask
    
    V_current = critic(cur_state)
    V_next = critic(next_state).detach()
    V_reward = torch.stack(rewards, dim=0).to(device).squeeze(1)

    # 关键改进：对于终止状态，V_next应该为0
    V_next_masked = V_next.squeeze(1) * (~done_mask)  # 终止状态的V_next设为0
    
    # 计算TD误差时使用masked V_next
    value_loss = torch.nn.functional.mse_loss(V_current.squeeze(1), V_reward + GAMMA * V_next_masked)
    # value_loss = value_loss1
    # if value_loss1.item()>100:
    #     print(1)

    # 计算 advantage loss (修改原始的policy loss)
    advantages = (V_reward + GAMMA * V_next_masked - V_current.squeeze(1)).detach()
    # advantages = V_reward + GAMMA * V_next.detach() - V_current.detach()
    # 这个detach很关键，防止梯度传到policy net, 确保policy和value net的独立性
    # normalized_reward = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    normalized_reward = advantages  # 实际用下来，不做normalize好像收敛更稳定


    log_probs = torch.stack(log_probs).squeeze(1)
    policy_loss = -(log_probs * normalized_reward).sum() / len(log_probs)
    # if i_episode < 50:
    #     policy_loss = policy_loss * 0
    
    # # 添加critic warmup阶段，前100个episode主要训练critic
    # if i_episode < 100:
    #     policy_loss = policy_loss * 0.1  # 减小policy loss，让critic先学好
    
    entropy_loss = torch.stack(entropies).sum()
    

    total_loss = policy_loss + 0.5 * value_loss - 0.001 * entropy_loss
    print(f"policy_loss: {policy_loss.item():.4f}, value_loss: {value_loss.item():.4f}, entropy_loss: {entropy_loss.item():.4f}, total_loss: {total_loss.item():.4f}")
    
    critic_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
    critic_optimizer.step()
    actor_optimizer.step()

    return total_loss.item()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

# 创建保存文件夹
save_dir = "save/Actor_Critic_TD"
os.makedirs(save_dir, exist_ok=True)

for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_loss_values = []  # 记录当前 episode 内所有优化步骤的loss

    for t in count():
        action, log_prob, entropy = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            # 保持next_state为真实的最终状态，而不是零向量
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # 减小惩罚，避免过于保守的策略
            reward = torch.tensor([-1], device=device)  # 从-5改为-1，减少过度惩罚
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward, log_prob, entropy, done)

        # Move to the next state
        state = next_state

        if done:
        # if done or len(memory) > 5:
            loss_val = optimize_model(i_episode)
            if done:
                if loss_val is not None:
                    episode_loss_values.append(loss_val)

                episode_durations.append(t + 1)
                # 如果当前 episode 有记录 loss，则计算平均 loss，否则记录为 None
                if episode_loss_values:
                    avg_loss = sum(episode_loss_values) / len(episode_loss_values)
                else:
                    avg_loss = None
                episode_losses.append(avg_loss)
                # 输出日志信息：episode编号、持续时间以及平均loss
                print(f"Episode {i_episode + 1}: Duration = {t + 1}, Average Loss = {avg_loss}")
                
                # 每20轮执行一次测试
                if (i_episode + 1) % test_interval == 0:
                    avg_test_duration = test_model(num_test_episodes=5)
                    test_episodes.append(i_episode + 1)
                    test_durations.append(avg_test_duration)
                    print(f"Test at Episode {i_episode + 1}: Average Duration = {avg_test_duration:.2f}")
                
                plot_durations()
                break

print("Complete")
# plot_durations(show_result=True)
plt.ioff()
plt.show()
plt.savefig(os.path.join(save_dir, "ActorCritic_training_and_test.png"), dpi=300, bbox_inches='tight')

# 执行最终测试
print("\nFinal Test:")
final_test_duration = test_model(num_test_episodes=10)
print(f"Final Test Result: Average Duration = {final_test_duration:.2f}")

# 保存测试结果到文件
test_results = {
    'test_episodes': test_episodes,
    'test_durations': test_durations,
    'final_test_duration': final_test_duration
}
import json
with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"Test results saved to {os.path.join(save_dir, 'test_results.json')}")

# 保存训练好的模型
actor_path = os.path.join(save_dir, "Actor_cartpole.pth")
critic_path = os.path.join(save_dir, "Critic_cartpole.pth")
torch.save(actor.state_dict(), actor_path)
torch.save(critic.state_dict(), critic_path)
print(f"Models saved as {actor_path} and {critic_path}")

# ----------------- 模型加载并生成视频演示 -----------------
# 创建新环境用于录制视频，这里使用 gymnasium 提供的 RecordVideo 封装器

# 注意：创建环境时需要指定 render_mode 为 "rgb_array" 以便录制视频帧
video_env = gym.make("CartPole-v1", render_mode="rgb_array")
# RecordVideo 会自动保存视频到指定目录
current_time = time.strftime("%Y%m%d_%H%M%S")
video_name = f"cartpole_ActorCritic_TD_{current_time}"
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
        prob = actor(state)
        action = prob.max(1).indices.view(1, 1)
    observation, reward, terminated, truncated, _ = video_env.step(action.item())
    done = terminated or truncated
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

video_env.close()
print("Video saved in 'videos' folder")
