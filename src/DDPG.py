import math
import random
from collections import deque
from collections import namedtuple
from itertools import count
import os

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import time


# 1. Initialize the environment
env = gym.make("Pendulum-v1")

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 2. Establish the Replay Memory
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 3. Establish the DDPG model
class DDPG_Actor(nn.Module):
    """Actor网络，用于生成动作"""

    def __init__(self, n_observations, n_actions):
        super(DDPG_Actor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer1_norm = nn.LayerNorm(128)
        self.layer2 = nn.Linear(128, 128)
        self.layer2_norm = nn.LayerNorm(128)
        self.layer3 = nn.Linear(128, n_actions)
        self.activation = nn.Tanh()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.gelu(self.layer1_norm(self.layer1(x)))
        x = F.gelu(self.layer2_norm(self.layer2(x)))
        return self.activation(self.layer3(x) * 2)


class DDPG_Critic(nn.Module):
    """Critic网络，用于评估动作的价值"""

    def __init__(self, n_observations, n_actions):
        super(DDPG_Critic, self).__init__()
        # self.layer1 = nn.Linear(n_observations + n_actions, 128)
        # self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, 1)
        self.state_layer = nn.Linear(n_observations, 128)
        self.state_norm = nn.LayerNorm(128)
        self.action_layer = nn.Linear(n_actions, 128)
        self.action_norm = nn.LayerNorm(128)
        self.layer2 = nn.Linear(256, 128)
        self.layer2_norm = nn.LayerNorm(128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = F.gelu(self.state_norm(self.state_layer(state)))
        y = F.gelu(self.action_norm(self.action_layer(action)))
        x = torch.cat((x, y), dim=1)
        x = F.gelu(self.layer2_norm(self.layer2(x)))
        return self.layer3(x)


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = torch.full(size, mu)
        self.reset()

    def reset(self):
        # 每个 episode 重置一次，让噪声重新回到均值
        self.state.fill_(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * torch.randn_like(self.state)
        self.state = self.state + dx
        return self.state


# 4. Training Hyperparameters and Helper Functions
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 3e-5
WARM_UP_STEPS = 10000
Value_LR = 1e-3
Init_noise_scale = 0.1
decay_rate = 0.999
ou_noise = OUNoise(size=env.action_space.shape, theta=0.15, sigma=0.5)

# Get number of actions from gym action space
n_actions = env.action_space.shape[0]
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DDPG_Actor(n_observations, n_actions).to(device)
target_policy_net = DDPG_Actor(n_observations, n_actions).to(device)
target_policy_net.load_state_dict(policy_net.state_dict())
value_net = DDPG_Critic(n_observations, n_actions).to(device)
target_value_net = DDPG_Critic(n_observations, n_actions).to(device)
target_value_net.load_state_dict(value_net.state_dict())

optimizer_actor = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
optimizer_critic = optim.AdamW(value_net.parameters(), lr=Value_LR, amsgrad=True, weight_decay=1e-4)

memory = ReplayMemory(10000)

steps_done = 0


def select_action(state, num_episodes):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        action_pred = policy_net(state)
    bs = action_pred.shape[0]

    action_max = env.action_space.high
    action_min = env.action_space.low
    # noise = torch.randn_like(state) * (action_max - action_min)
    noise = (
        torch.empty(action_pred.shape).uniform_(-1, 1)
        * (action_max - action_min)
        * Init_noise_scale
        * decay_rate ** (steps_done / 10000)
    ).cuda()
    # noise = ou_noise.sample().unsqueeze(0).repeat(bs, 1).cuda()
    return action_pred + noise


episode_durations = []
# 全局记录每个 episode 的平均 loss
episode_losses = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# 5. Training Function
def optimize_model():
    if len(memory) < WARM_UP_STEPS:
        return None, None
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # Value Net Loss (Bellman Equation)
    next_state_action_prediction = target_policy_net(next_state_batch).detach()
    next_value_prediction = target_value_net(next_state_batch, next_state_action_prediction).detach()  # [B,1]
    cur_value_prediction = value_net(state_batch, action_batch)  # [B,1]
    expected_state_action_values = (next_value_prediction * GAMMA) + reward_batch.unsqueeze(1)
    criterion = nn.SmoothL1Loss()
    loss = criterion(cur_value_prediction.squeeze(1), expected_state_action_values.squeeze(1))  # * 50

    # Optimize the Value Net
    optimizer_critic.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(value_net.parameters(), 1)
    optimizer_critic.step()

    # Actor Net Loss (Policy Gradient)
    # Use a new forward pass through value_net to create a separate computation graph

    cur_state_action_prediction = policy_net(state_batch)  # [B,1]
    action_loss = -target_value_net(state_batch, cur_state_action_prediction).mean()

    # Optimize the Policy Net
    optimizer_actor.zero_grad()
    action_loss.backward()
    # torch.nn.utils.clip_grad_value_(value_net.parameters(), 100)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer_actor.step()

    return loss.item(), action_loss.item()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 300
else:
    num_episodes = 50

# 创建保存文件夹
save_dir = "save/DDPG"
os.makedirs(save_dir, exist_ok=True)

for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_loss_values = []  # 记录当前 episode 内所有优化步骤的loss
    episode_action_loss_values = []
    reward_sum = 0
    ou_noise.reset()
    ou_noise.sigma = max(0.2, ou_noise.sigma * decay_rate)

    for t in count():
        action = select_action(state, num_episodes)
        observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        reward_sum += reward
        reward = torch.tensor(np.array([reward]), device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).squeeze(-1)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        loss_val, action_loss_val = optimize_model()
        if loss_val is not None:
            episode_loss_values.append(loss_val)
            episode_action_loss_values.append(action_loss_val)
        # Soft update of the target networks' weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_value_net_state_dict = target_value_net.state_dict()
        value_net_state_dict = value_net.state_dict()
        for key in value_net_state_dict:
            target_value_net_state_dict[key] = value_net_state_dict[key] * TAU + target_value_net_state_dict[key] * (
                1 - TAU
            )
        target_value_net.load_state_dict(target_value_net_state_dict)

        target_policy_net_state_dict = target_policy_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_policy_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_policy_net_state_dict[key] * (
                1 - TAU
            )
        target_policy_net.load_state_dict(target_policy_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            # 如果当前 episode 有记录 loss，则计算平均 loss，否则记录为 None
            if episode_loss_values:
                avg_loss = sum(episode_loss_values) / len(episode_loss_values)
            else:
                avg_loss = None
            if episode_action_loss_values:
                avg_action_loss = sum(episode_action_loss_values) / len(episode_action_loss_values)
            else:
                avg_action_loss = None
            episode_losses.append(avg_loss)
            # 输出日志信息：episode编号、持续时间以及平均loss
            print(
                f"Episode {i_episode + 1}: Duration = {t + 1}, Average Loss = {avg_loss}, Average Action Loss = {avg_action_loss}, Reward Sum = {reward_sum}"
            )
            plot_durations()
            break

print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()
plt.savefig(os.path.join(save_dir, "DDPG_pendulum.png"))

# 保存训练好的模型
model_path = os.path.join(save_dir, "DDPG_pendulum.pth")
torch.save(policy_net.state_dict(), model_path)
print(f"Model saved as {model_path}")

# ----------------- 模型加载并生成视频演示 -----------------
# 创建新环境用于录制视频，这里使用 gymnasium 提供的 RecordVideo 封装器

# 注意：创建环境时需要指定 render_mode 为 "rgb_array" 以便录制视频帧
video_env = gym.make("Pendulum-v1", render_mode="rgb_array")
# RecordVideo 会自动保存视频到指定目录
current_time = time.strftime("%Y%m%d_%H%M%S")
video_name = f"pendulum_DDPG_{current_time}"
video_env = RecordVideo(
    video_env, 
    video_folder="videos",
    name_prefix=video_name,
    episode_trigger=lambda episode_id: True
)

# 加载保存的模型
policy_net.load_state_dict(torch.load(model_path, map_location=device))
policy_net.eval()  # 切换到评估模式

# 初始化环境并开始录制视频
state, info = video_env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False

while not done:
    # 根据当前状态选择动作（贪婪策略）
    with torch.no_grad():
        # action = policy_net(state).max(1).indices.view(1, 1)
        action = policy_net(state)
    observation, reward, terminated, truncated, _ = video_env.step(action[0].cpu().numpy())
    done = terminated or truncated
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

video_env.close()
print("Video saved in 'videos' folder")
