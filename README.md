# DeepQLearningCartPole
This project is to train an agent to solve the CartPole problem in OpenAI Gym by implementing a Deep Q Network.

### Aims
Compare 4 scenarios:  
1. Q-learning (no experience replay and no target network)  
2. Q-learning with experience replay (no target network)  
3. Q-learning with a target network (no experience replay)  
4. Q-learning with experience replay and a target network  

### Network Structure 
Input layer of 4 nodes (corresponding to the 4 state features)  
Two hidden layers of 10 rectified linear units (fully connected)  
Output layer of 2 identity units (fully connected) that compute the Q-values of the two actions

### Train this neural network by gradient Q-learning
Discount factor: gamma=0.99  
Exploration strategy: epsilon-greedy with epsilon=0.05  
learingRate = 0.1  
Maximum horizon of 500 steps per episode  
Train for a maximum of 1000 episodes  


### Result
The result images are put in resultIMG folder.  
Q-learning with experience replay and a target network quickly learns to balance the pole well. The target network ensures that learning is done gradually and in a stable fashion. By using a target network, Q-learning updates the network only periodically, similar to what is done in value
iteration. Experience replay helps to break correlations between updates and speeds up learning since the network repeatedly learns from the same experiences. This reduces the number of episodes that are needed to find a good policy.  
Without the target network and without any experience reply, the cart is simply incapable of 4 learning a good policy to balance the pole within 1000 episodes. This can be observed by the fact that very little reward is earned in each episode and there is no noticeable improvement.  
With a target network, but no experience replay, the cart manages to balance the pole well a few times, but this takes many episode and the network is clearly unstable. The absence of experience replay prevents the network from learning quickly and the correlations among subsequent updates induce instability.  
With experience replay, but no target network, the cart often balances the pole for many steps, but the network is clearly unstable. Without a target network, consecutive updates create instability in the network, which explains why it does not consistently improve.