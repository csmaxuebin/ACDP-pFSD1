# 用于连续动作的PPO
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state



# ------------------------------------- #
# 策略网络--输出连续动作的高斯分布的均值和标准差
# ------------------------------------- #

class GaussianPolicyNet(nn.Module):
	def __init__(self, n_states, n_hiddens, n_actions):
		super(GaussianPolicyNet, self).__init__()
		self.fc1 = nn.Linear(n_states, n_hiddens)
		self.fc_mu = nn.Linear(n_hiddens, n_actions)
		self.fc_std = nn.Linear(n_hiddens, n_actions)
	# 前向传播
	def forward(self, x):  #
		x = self.fc1(x.float())  # [b, n_states] --> [b, n_hiddens]
		x = F.relu(x)
		mu = self.fc_mu(x)  # [b, n_hiddens] --> [b, n_actions]
		#mu = 2 * torch.tanh(mu)  # 值域 [-2,2]
		mu = torch.tanh(mu)
		std = self.fc_std(x)  # [b, n_hiddens] --> [b, n_actions]
		std = F.softplus(std)  # 值域 小于0的部分逼近0，大于0的部分几乎不变
		return mu, std

class BetaPolicyNet(nn.Module):
	def __init__(self, n_states, n_hiddens, n_actions):
		super(BetaPolicyNet, self).__init__()
		self.l1 = nn.Linear(n_states, n_hiddens)
		self.l2 = nn.Linear(n_hiddens, n_hiddens)
		self.alpha_head = nn.Linear(n_hiddens, n_actions)
		self.beta_head = nn.Linear(n_hiddens, n_actions)

	def forward(self, x):
		x = self.l1(x.float())
		a = torch.tanh(x)
		a = torch.tanh(self.l2(a))
		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0
		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	# def get_alpha(self,state):
	# 	alpha,beta = self.forward(state)
	# 	#dist = Beta(alpha, beta)
	# 	return alpha,beta

	def deterministic_act(self, state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

# ------------------------------------- #
# 价值网络 -- 评估当前状态的价值
# ------------------------------------- #

class ValueNet(nn.Module):
	def __init__(self, n_states, n_hiddens):
		super(ValueNet, self).__init__()
		self.fc1 = nn.Linear(n_states, n_hiddens)
		self.fc2 = nn.Linear(n_hiddens, 1)
	# 前向传播
	def forward(self, x):
		x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
		x = F.relu(x)
		x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
		return x

# ------------------------------------- #
# 模型构建--处理连续动作
# ------------------------------------- #

class PPO:
	def __init__(self, n_states, n_actions, device):
		self.device = device
		self.noise = OUNoise(n_actions)

		# 实例化策略网络
		if self.policy == "Gaussian":
			self.actor = GaussianPolicyNet(n_states, self.n_hiddens, n_actions).to(self.device)
		else:
			self.actor = BetaPolicyNet(n_states, self.n_hiddens, n_actions).to(self.device)
		# 实例化价值网络
		self.critic = ValueNet(n_states, self.n_hiddens).to(self.device)
		# 策略网络的优化器
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		# 价值网络的优化器
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

	# def get_beta(self,state):
	# 	alpha,beta=self.actor.get_alpha(state)
	# 	return alpha,beta

	# 动作选择
	def take_action(self, state):  # 输入当前时刻的状态
		# [n_states]-->[1,n_states]-->tensor
		state = torch.tensor(state[np.newaxis, :]).to(self.device)
		if self.policy == "Gaussian":
			# 预测当前状态的动作，输出动作概率的高斯分布
			mu, std = self.actor(state)
			# 构造高斯分布
			action_dict = torch.distributions.Normal(mu, std)
			# 随机选择动作
			action = action_dict.sample().item()
			action = np.clip(action, -1, 1)  # 确保action的值在[-1, 1]范围内
		else:
			# mu, sigma_sq = self.model(Variable(state).cuda())
			alpha,beta = self.actor(state)
			action_dict = torch.distributions.Beta(alpha,beta)
			sample = action_dict.sample()
			action = sample.item()
			#action = (sample * 2 - 1).item()  # 定义域[-1,1]
			#action = (sample * 5 ).item()  # 定义域[0,1.5]
			noise = self.noise.sample()
			#action = action + noise
			log_prob = action_dict.log_prob(sample)
			entropy = action_dict.entropy()
		return [action]  # 返回动作值

	# 训练
	def update(self, transition_dict):
		# 提取数据集
		states = torch.tensor(transition_dict['states'], dtype=torch.float).view(-1, 3).to(self.device)  # [b,n_states]
		actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).view(-1, 3).to(self.device)  # [b,n_states]
		#dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]

		# 价值网络--目标，获取下一时刻的state_value  [b,n_states]-->[b,1]
		next_states_target = self.critic(next_states)
		# 价值网络--目标，当前时刻的state_value  [b,1]
		td_target = rewards + self.gamma * next_states_target * 1
		# 价值网络--预测，当前时刻的state_value  [b,n_states]-->[b,1]
		td_value = self.critic(states)
		# 时序差分，预测值-目标值  # [b,1]
		td_delta = td_value - td_target

		# 对时序差分结果计算GAE优势函数
		td_delta = td_delta.cpu().detach().numpy()  # [b,1]
		advantage_list = []  # 保存每个时刻的优势函数
		advantage = 0  # 优势函数初始值
		# 逆序遍历时序差分结果，把最后一时刻的放前面
		for delta in td_delta[::-1]:
			advantage = self.gamma * self.lmbda * advantage + delta
			advantage_list.append(advantage)
		# 正序排列优势函数
		advantage_list.reverse()
		# numpy --> tensor
		advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
		#高斯分布
		# 策略网络--预测，当前状态选择的动作的高斯分布
		if self.policy == "Gaussian":
			mu, std = self.actor(states)  # [b,1]
			# 基于均值和标准差构造正态分布
			action_dists = torch.distributions.Normal(mu.detach(), std.detach())
			# 从正态分布中选择动作，并使用log函数
			old_log_prob = action_dists.log_prob(actions)
		else:
			alpha, beta = self.actor(states)
			action_dists = torch.distributions.Beta(alpha.detach(), beta.detach())
			old_log_prob = action_dists.log_prob(actions)

		# 一个序列训练epochs次
		for _ in range(self.epochs):
			# # 预测当前状态下的动作
			if self.policy == "Gaussian":
				mu, std = self.actor(states)
				# 构造正态分布
				action_dists = torch.distributions.Normal(mu, std)
			else:
				alpha, beta = self.actor(states)
				action_dists = torch.distributions.Beta(alpha, beta)
			# 当前策略在 t 时刻智能体处于状态 s 所采取的行为概率
			log_prob = action_dists.log_prob(actions)
			# 计算概率的比值来控制新策略更新幅度
			ratio = torch.exp(log_prob - old_log_prob)

			# 公式的左侧项
			surr1 = ratio * advantage
			# 公式的右侧项，截断
			surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)

			# 策略网络的损失PPO-clip
			actor_loss = torch.mean(-torch.min(surr1,surr2))
			# 价值网络的当前时刻预测值，与目标价值网络当前时刻的state_value之差
			critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

			# 优化器清0
			self.actor_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()
			# 梯度反传
			actor_loss.backward()
			critic_loss.backward()
			# 参数更新
			self.actor_optimizer.step()
			self.critic_optimizer.step()

		return actor_loss.item(),critic_loss.item()
