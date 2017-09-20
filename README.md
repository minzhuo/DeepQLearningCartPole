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