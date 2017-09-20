import numpy as np
import gym
import tensorflow as tf 
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

class DQN:
	def __init__(self, size_action, size_state):
		self.cost = []
		self.learning_rate = 0.01
		self.epsilon_greedy = 0.05
		self.discount_factor = 0.99
		self.size_replay = 1000
		self.size_mini_batch = 50
		self.size_action = size_action
		self.size_state = size_state
		self.memory_index = 0
		self.learn_step_counter = 0
		self.memory = np.zeros((self.size_replay , size_state * 2 + 2))

		self.Build_EQNN()
		self.Build_TNN()
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def Build_EQNN(self):

		w_initial = tf.random_normal_initializer(0., 0.4)
		b_initail = tf.constant_initializer(0.2) 

		self.state = tf.placeholder(tf.float32, [None, self.size_state], name='state') 
		self.true_lable = tf.placeholder(tf.float32, [None, self.size_action], name='Q_target')
		with tf.variable_scope('Enetwork'):
			name  = ['e_net', tf.GraphKeys.GLOBAL_VARIABLES]
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.size_state, 10], initializer=w_initial, collections=name)
				b1 = tf.get_variable('b1', [1, 10], initializer=b_initail , collections=name)
				l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)
			
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [10, 10], initializer=w_initial, collections=name)
				b2 = tf.get_variable('b2', [1, 10], initializer=b_initail , collections=name)
				l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

			with tf.variable_scope('l3'):
				w3 = tf.get_variable('w3', [10, self.size_action], initializer=w_initial, collections=name)
				b3 = tf.get_variable('b3', [1, self.size_action], initializer=b_initail , collections=name)
				self.q_e = tf.matmul(l2, w3) + b3

			with tf.variable_scope('loss'):
				self.loss = tf.reduce_mean(tf.squared_difference(self.true_lable, self.q_e))
			with tf.variable_scope('train'):
				self.train= tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
		
	def Build_TNN(self):

		w_initial = tf.random_normal_initializer(0., 0.4)
		b_initail = tf.constant_initializer(0.2) 
		self.state1 = tf.placeholder(tf.float32, [None, self.size_state], name='state1') 
		with tf.variable_scope('target_net'):
			name = ['t_net', tf.GraphKeys.GLOBAL_VARIABLES]

			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.size_state, 10], initializer=w_initial, collections=name)
				b1 = tf.get_variable('b1', [1, 10], initializer=b_initail , collections=name)
				l1 = tf.nn.relu(tf.matmul(self.state1, w1) + b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [10, 10], initializer=w_initial, collections=name)
				b2 = tf.get_variable('b2', [1, 10], initializer=b_initail , collections=name)
				l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

         
			with tf.variable_scope('l3'):
				w3 = tf.get_variable('w3', [10, self.size_action], initializer=w_initial, collections=name)
				b3 = tf.get_variable('b3', [1, self.size_action], initializer=b_initail , collections=name)
				self.q_t = tf.matmul(l2, w3) + b3

	def copy_replace(self):
		t_params = tf.get_collection('t_net')
		e_params = tf.get_collection('e_net')
		self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

	def memory_store(self, s, a, r, s_):
		transition = np.hstack((s, [a, r], s_))
		index = self.memory_index % self.size_replay
		self.memory[index, :] = transition
		self.memory_index += 1

	def nextaction(self, ob):
		action = 0
		ob = ob[np.newaxis, :]

		if np.random.uniform(0,1) < self.epsilon_greedy :
			action = np.random.randint(0, self.size_action)
		else:
			Q_action = self.sess.run(self.q_e, feed_dict = {self.state : ob})
			action = np.argmax(Q_action)

		return action

	def learn1(self, ob, ob_, action, reward):
		ob = ob[np.newaxis, :]
		ob_ = ob_[np.newaxis, :]
		true_lable, q_e = self.sess.run([self.q_e, self.q_e],feed_dict={self.state: ob_,  self.state: ob,  })
		q_target = q_e.copy()
		batch_index = 0
		eval_act_index = action
		q_target[0, action] = reward + self.discount_factor * np.max(true_lable, axis=1)
		_, cost = self.sess.run([self.train, self.loss], feed_dict={self.state: ob, self.true_lable: q_target})
		self.cost.append(cost)
		self.learn_step_counter += 1

	def learn2(self):
		if self.memory_index > self.size_replay:
			s_index = np.random.choice(self.size_replay, size=self.size_mini_batch)
		else:
			s_index = np.random.choice(self.memory_index, size=self.size_mini_batch)
		mini_batch_memory = self.memory[s_index, :]

		true_lable, q_e = self.sess.run([self.q_e, self.q_e],feed_dict={self.state: mini_batch_memory[:, -self.size_state:], self.state: mini_batch_memory[:, :self.size_state], })

		q_target = q_e.copy()

		batch_index = np.arange(self.size_mini_batch, dtype=np.int32)
		act_index = mini_batch_memory[:, self.size_state].astype(int)
		reward = mini_batch_memory[:, self.size_state + 1]

		q_target[batch_index, act_index] = reward + self.discount_factor * np.max(true_lable, axis=1)
		_, cost = self.sess.run([self.train, self.loss],feed_dict={self.state: mini_batch_memory[:, :self.size_state],self.true_lable: q_target})
		self.cost.append(cost)

	def learn3(self, ob, ob_, action, reward):
		ob = ob[np.newaxis, :]
		ob_ = ob_[np.newaxis, :]

		true_lable, q_e = self.sess.run([self.q_t, self.q_e], feed_dict={ self.state1: ob_, self.state: ob, })
		q_target = q_e.copy()
		q_target[0, action] = reward + self.discount_factor * np.max(true_lable, axis=1)
		_, cost = self.sess.run([self.train, self.loss],feed_dict={self.state: ob,self.true_lable: q_target})
		self.cost.append(cost)

	def learn4(self, ob, ob_, action, reward):

			s_index = np.random.choice(self.memory_size, size=self.size_mini_batch)
		else:
			s_index = np.random.choice(self.memory_index, size=self.size_mini_batch)
		mini_batch_memory = self.memory[s_index, :]

		true_lable, q_e = self.sess.run([self.q_t, self.q_e],feed_dict={self.state1: mini_batch_memory[:, -self.size_state:],  
			self.state: mini_batch_memory[:, :self.size_state], })

		q_target = q_e.copy()

		batch_index = np.arange(self.size_mini_batch, dtype=np.int32)
		act_index = mini_batch_memory[:, self.size_state].astype(int)
		reward = mini_batch_memory[:, self.size_state + 1]

		q_target[batch_index, act_index] = reward + self.discount_factor * np.max(true_lable, axis=1)
		_, cost = self.sess.run([self.train, self.loss],feed_dict={self.state: mini_batch_memory[:, :self.size_state],
                                                self.true_lable: q_target})
		self.cost.append(cost)

	def print_cost():
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training Times')
		plt.show()

	def print_reward(total_reward):
		fig = plt.figure(1, [5,4])
		ax = fig.add_subplot(111)
		ax.plot( np.arange(len(reward)) , reward ,  color='c', linewidth=2)
		ax.set_xscale('log')
		plt.ylabel('Total Reward')
		plt.xlabel('Episode')
		ax.xaxis.set_major_locator(LogLocator(base = 10.0))
		plt.show()

def reward_compate(ob, env):
	x, x_dot, theta, theta_dot = ob
	r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
	r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
	return r1 + r2

def scenarios_1(env):
	dqn = 	DQN(env.action_space.n, env.observation_space.shape[0])
	total_steps = 0
	total_reward = []

	for i_episode in range(1000) :
		ob = env.reset()
		expecttion_reward = 0

	  	while True:
			env.render()

			action = dqn.nextaction(ob)
			ob_, reward, done, _ = env.step(action)
			reward = reward_compate(ob_, env)

			expecttion_reward += reward
			dqn.learn1(ob, ob_, action, reward)

			if done or total_steps > 500:
				break

			ob = ob_
			total_steps += 1

	return total_reward

def scenarios_2(env):
	dqn = DQN(env.action_space.n, env.observation_space.shape[0])
	total_steps = 0
	total_reward = []

	for i_episode in range(1000) :
		ob = env.reset()
		expecttion_reward = 0

	  	while True:
			env.render()
			action = dqn.nextaction(ob)
			ob_, reward, done, _ = env.step(action)
			reward = reward_compate(ob_, env)
			dqn.memory_store(ob, action, reward, ob_)
			expecttion_reward += reward
			dqn.learn2()

			if done or total_steps > 500:
				break

			ob = ob_
			total_steps += 1
	return total_reward

def scenarios_3(env):
	dqn = DQN(env.action_space.n, env.observation_space.shape[0])
	total_steps = 0
	total_reward = []

	for i_episode in range(1000) :
		if i_episode % 2 == 0 :
			dqn.copy_replace()
		ob = env.reset()
		expecttion_reward = 0

	  	while True:
			env.render()
			action = dqn.nextaction(ob)
			ob_, reward, done, _ = env.step(action)
			reward = reward_compate(ob_, env)
			expecttion_reward += reward
			dqn.learn3(ob, ob_, action, reward)
			if done or total_steps > 500:
				break

			ob = ob_
			total_steps += 1
	return total_reward

def scenarios_4(env):
	dqn = DQN(env.action_space.n, env.observation_space.shape[0])
	total_steps = 0
	total_reward = []

	for i_episode in range(1000) :
		if i_episode % 2 == 0 :
			dqn.copy_replace()
		ob = env.reset()
		expecttion_reward = 0

	  	while True:
	  		env.render()
			action = dqn.nextaction(ob)
			ob_, reward, done, _ = env.step(action)
			reward = reward_compate(ob_, env)
			dqn.memory_store(ob, action, reward, ob_)
			expecttion_reward += reward
			dqn.learn4(ob, ob_, action, reward)
			if done or total_steps > 500:
				break

			ob = ob_
			total_steps += 1
	return total_reward

def main():
	np.random.seed(1)
	tf.set_random_seed(1)
	env = gym.make('CartPole-v0')
	env = env.unwrapped
	#scenarios_1(env)
	#scenarios_2(env)
	#scenarios_3(env)
	scenarios_4(env)

if __name__ == "__main__":
	main()
