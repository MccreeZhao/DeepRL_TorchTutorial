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
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "log_prob", "entropy"))


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
critic_lr = 1e-3
actor_lr = 1e-5

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
critic_optimizer = optim.AdamW(critic.parameters(), lr=critic_lr, amsgrad=True)
actor_optimizer = optim.AdamW(actor.parameters(), lr=actor_lr, amsgrad=True)
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
def optimize_model(i_episode):

    accumulated_reward = []
    log_probs = []
    advantages = []
    cur_state = []
    next_state = []
    rewards = []
    entropies = []
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
    # accumulated_reward = torch.tensor(accumulated_reward, device=device, dtype=torch.float)
    # accumulated_reward = (accumulated_reward - accumulated_reward.mean()) / (accumulated_reward.std() + 1e-5)

    # 计算 value loss
    
    cur_state = torch.stack(cur_state, dim=0).to(device).squeeze(1)
    next_state = torch.stack(next_state, dim=0).to(device).squeeze(1)
    not_terminate_mask = next_state.sum(dim=1) != 0
    
    V_current = critic(cur_state)
    V_next = critic(next_state).detach()
    V_reward = torch.stack(rewards, dim=0).to(device).squeeze(1)

    # value_loss1 = torch.nn.functional.mse_loss(V_current.squeeze(1)* not_terminate_mask, V_reward* not_terminate_mask + GAMMA * V_next.squeeze(1).detach() * not_terminate_mask)
    value_loss = torch.nn.functional.mse_loss(V_current.squeeze(1), V_reward + GAMMA * V_next.squeeze(1) * not_terminate_mask)
    # value_loss = value_loss1
    # if value_loss1.item()>100:
    #     print(1)

    # 计算 advantage loss (修改原始的policy loss)
    advantages = (V_reward + GAMMA * V_next.squeeze(1) - V_current.squeeze(1)).detach() * not_terminate_mask
    # advantages = V_reward + GAMMA * V_next.detach() - V_current.detach()
    # 这个detach很关键，防止梯度传到policy net, 确保policy和value net的独立性
    # normalized_reward = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    normalized_reward = advantages  # 实际用下来，不做normalize好像收敛更稳定

    log_probs = torch.stack(log_probs).squeeze(1)
    policy_loss = -(log_probs * normalized_reward).sum() / len(log_probs)
    # if i_episode < 50:
    #     policy_loss = policy_loss * 0
    entropy_loss = torch.stack(entropies).sum()
    

    total_loss = policy_loss + 0.5 * value_loss - 0.001 * entropy_loss
    print(f"policy_loss: {policy_loss.item()}, value_loss: {value_loss.item()}, entropy_loss: {entropy_loss.item()}, total_loss: {total_loss.item()}")
    
    critic_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1)
    critic_optimizer.step()
    actor_optimizer.step()

    return total_loss.item()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 3000
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
            next_state = state*0
            reward = torch.tensor([-5], device=device)
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward, log_prob, entropy)

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
                plot_durations()
                break

print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()
plt.savefig(os.path.join(save_dir, "ActorCritic_cartpole.png"))

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
