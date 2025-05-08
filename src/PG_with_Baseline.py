import math
import random
from collections import deque
from collections import namedtuple
from itertools import count
import time
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


# 3. Establish the Dueling DQN model
# class PG_BS_Reinforce(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(PG_BS_Reinforce, self).__init__()

#         # Shared layers
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)

#         # Prob stream (P(a|s))
#         self.prob_stream = nn.Linear(128, n_actions)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))

#         return self.softmax(self.prob_stream(x))


# class Value_Net(nn.Module):
#     def __init__(self, n_observations):
#         super(Value_Net, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.value_stream = nn.Linear(128, 1)

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.value_stream(x)


class ActorCritic(nn.Module):
    """共享两层 MLP 的 Actor‑Critic 网络。"""

    def __init__(self, num_obs: int, num_actions: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.shared(x)
        return self.softmax(self.policy_head(x)), self.value_head(x)


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
LR = 3e-3

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# policy_net = PG_BS_Reinforce(n_observations, n_actions).to(device)
# value_net = Value_Net(n_observations).to(device)
actor_critic = ActorCritic(n_observations, n_actions).to(device)
optimizer = optim.AdamW(actor_critic.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000)
memory.clear()
steps_done = 0


def select_action(state):
    global steps_done
    # sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    # if sample > eps_threshold:
    #     with torch.no_grad():
    #         # t.max(1) will return the largest column value of each row.
    #         # second column on max result is index of where max element was
    #         # found, so we pick action with the larger expected reward.
    #         return policy_net(state).max(1).indices.view(1, 1)
    # else:
    #     return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    prob, _ = actor_critic(state)
    # log_prob = torch.log(prob)
    # action = torch.multinomial(prob, num_samples=1)
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
def optimize_model():

    accumulated_reward = []
    log_probs = []
    advantages = []
    cur_state = []
    next_state = []
    rewards = []
    entropies = []
    reward_sum = 0
    while len(memory) > 0:
        # item = memory.popleft()
        item = memory.popright()
        reward_sum = reward_sum * GAMMA + item.reward
        accumulated_reward.insert(0, reward_sum)
        log_probs.insert(0, item.log_prob)
        cur_state.insert(0, item.state)
        next_state.insert(0, item.next_state)
        rewards.insert(0, item.reward)
        entropies.insert(0, item.entropy)
    accumulated_reward = torch.tensor(accumulated_reward, device=device, dtype=torch.float)
    accumulated_reward = (accumulated_reward - accumulated_reward.mean()) / (accumulated_reward.std() + 1e-5)

    # 计算 value loss
    cur_state = torch.stack(cur_state, dim=0).to(device).squeeze(1)
    # next_state = torch.stack(next_state[0:-1], dim=0).to(device).squeeze(1)
    _, V_current = actor_critic(cur_state)
    # V_next = value_net(next_state)
    # V_reward = torch.stack(rewards, dim=0).to(device).squeeze(1)[0:-1]

    # value_loss = torch.nn.functional.mse_loss(V_current.squeeze(1), V_reward + GAMMA * V_next.squeeze(1))
    value_loss = torch.nn.functional.mse_loss(V_current.squeeze(1), accumulated_reward)

    # 计算 advantage loss (修改原始的policy loss)
    advantages = accumulated_reward - V_current.squeeze(1).detach()
    # advantages = V_reward + GAMMA * V_next.detach() - V_current.detach()
    # 这个detach很关键，防止梯度传到policy net, 确保policy和value net的独立性
    # normalized_reward = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    normalized_reward = advantages  # 实际用下来，不做normalize好像收敛更稳定

    log_probs = torch.stack(log_probs).squeeze(1)
    loss = (log_probs * normalized_reward).sum()
    # print(f"loss.max(): {loss.max()}, loss.min(): {loss.min()}")
    loss = -loss / len(log_probs)

    # entropy_loss = torch.distributions.Categorical(logits=log_probs).entropy()
    entropy_loss = torch.stack(entropies).sum()

    loss = loss + 0.02 * value_loss + 0.00001 * entropy_loss

    print(f"loss: {loss.item()}, value_loss: {value_loss.item()}, entropy_loss: {entropy_loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(actor_critic.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.2)
    optimizer.step()

    return loss.item()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# 创建保存文件夹
save_dir = "save/PG_with_Baseline"
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
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward, log_prob, entropy)

        # Move to the next state
        state = next_state

        if done:
            loss_val = optimize_model()
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
plt.savefig(os.path.join(save_dir, "PG_BS_Reinforce_cartpole.png"))

# 保存训练好的模型
model_path = os.path.join(save_dir, "PG_BS_cartpole.pth")
torch.save(actor_critic.state_dict(), model_path)
print(f"Model saved as {model_path}")

# ----------------- 模型加载并生成视频演示 -----------------
# 创建新环境用于录制视频，这里使用 gymnasium 提供的 RecordVideo 封装器

# 注意：创建环境时需要指定 render_mode 为 "rgb_array" 以便录制视频帧
video_env = gym.make("CartPole-v1", render_mode="rgb_array")
# RecordVideo 会自动保存视频到指定目录
current_time = time.strftime("%Y%m%d_%H%M%S")
video_name = f"cartpole_PG_with_Baseline_{current_time}"
video_env = RecordVideo(
    video_env, 
    video_folder="videos",
    name_prefix=video_name,
    episode_trigger=lambda episode_id: True
)

# 加载保存的模型
actor_critic.load_state_dict(torch.load(model_path, map_location=device))
actor_critic.eval()

# 初始化环境并开始录制视频
state, info = video_env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False

while not done:
    # 根据当前状态选择动作（贪婪策略）
    with torch.no_grad():
        action = actor_critic(state)[0].max(1).indices.view(1, 1)
    observation, reward, terminated, truncated, _ = video_env.step(action.item())
    done = terminated or truncated
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

video_env.close()
print("Video saved in 'videos' folder")
