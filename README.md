# COVID-19-Deep-Reinforcement-Learning-Algorithm-for-Case-Prediction
This repository is a code adapted to account for general control implementations and how these would affect the rise and fall of infections using deep reinforcement learning (DRL). The code is adapted from a DRL stock prediction tutorial, and data is provided by the state of California. 

## Code & Data Sources
The code is adapted from https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/, with data gathered from https://data.chhs.ca.gov/dataset/covid-19-time-series-metrics-by-county-and-state/resource/046cdd2b-31e5-4d34-9ed3-b48cdbc4be7a?view_id=eb8a870d-78a9-4a3a-bb9e-aef74924085c.

The CSV files provide data as follows:
- covid19cases_test - All cases and deaths recorded daily from February 2020 until most current, across all counties in California (read more on the data's website for changes in data recording).
- covidla - All cumulative cases and deaths for only the county of Los Angeles.
- covidla_ - A subset of all cumulative cases and deaths for the county of Los Angeles.


## Reinforcement Learning
Reinforcement learning is a branch of machine learning in which and algorithm, or agent, interacts with its surrounding environment to receive feedback throughout training. The agent learns through this feedback loop of positive or negative rewards, attempting to create a policy, or a model of a sequence of choices, in which would maximize its total rewards. Such a policy is known as an optimal policy. This policy is calculated by a value function which determines the value of each action or state-action pair. The agent's setting or condition is its state, and its choice of how to move to the next state is its action. The value function utilizes the agents' state, next state, action, and reward to determine the value or quality of each action or state-action pair. This function is further discounted to grant greater weight to future rewards rather than immediate gratification. As the agent steps through its environment and makes subsequent decisions, it experiences and observes various combinations of states and actions, leading to very different rewards. 

![image](https://github.com/kimk333/COVID-19-Deep-Reinforcement-Learning-Algorithm-for-Case-Prediction/assets/109542237/82503423-1273-4e62-827b-f0441bfcaa3c)

Additionally, unlike supervised or unsupervised learning, reinforcement learning does not rely on labeled data and is able to efficiently learn with less historical data. As seen with Google DeepMind's AlphaGo, reinforcement learning agents are even able to surpass many humans and traditional algorithms by learning through trial-and error. 

The general guidelines for the framework of a reinforcement learning algorithm are as follows: 
1. Import all neccessary libraries and packages
2. Create the agent who makes all the decisions (single-agent RL)
3. Define basic functions for formatting the values, sigmoid function, reading the data file, etc...
4. Train the agent
5. Evaluate the agent's performance and update policy

## Deep Reinforcement Learning
However, as the problem becomes more complex such as an increase in states, infinitely many states, or an increase in dimensionality, the value function may be very difficult (even impossible) to calculate. To resolve this, neural networks are incorporated into the algorithm to handle such complex computations.


## Defining our Problem
For this adaptation, the problem is defined as the following for case prediction:
- Environment: community, LA county
- Agent: an agent A that works in an Environment E, in this case, the government
- Actions: nothing, implement complete lockdown, or implement complete lockdown and a 100% effective and immediate cure (3)
- States: Data values
- Rewards: recovered/cured increases (+)
For this implementation, the actions are rather naive and less realistic. Further research and testing may be incorporated to make the actions more realistic.

## Our Policy & Training Algorithm
For our policy, we utilize an epsilon-greedy policy. This policy is defined by a variable, epsilon, which defines how exploratory the agents moves are. In reinforcement learning, it is critical to find and effective balance between exploratory moves (completely random) and exploitative moves (following the choices already made that the agent knows will give it the largest reward it has already discovered). In finding this balance, the agent finds new combinations which may lead to greater rewards than previously observed, while still following its currently known policy to keep its increase in rewards consistent. Epsilon may also decline over the training process to gradually reduce exploration as the agent learns an optimal policy for better convergence. 

The policy is followed by a deep reinforcement learning algorithm known as Deep Q-Learning (DQN). This algorithm utilizes a deep neural network structure to calculate the Q-value function, or the state-action pair function for the agent. The function finds optimal policy by maximizing the expected value of the total reward R over any successive steps, from the current stateThis algorithm helps the agent be able to effectively train through complex environments. Additionally, the algorithm is model-free, hence having no need for the agent to have a complete model (if at all) of its environment to train. With the incorporation of neural networks, the algorithm does not need to memorize every state-action pair, greatly reducing computational complexity, and allowing computation for more complex environments.

## How to Run the Code
- func.py - All the basic functions needed to calculate format the data, extract the data, and calculate the next state.
- DQN.py - The DQN-based agent defined with hyperparameters such as the discount gamma and epsilon values; along with the act() and expReplay() methods used to predict the next action and reset the memory as the agent learns based off its experience, respectively
- main.py - The main training code and loop; depending on the action that is predicted by the model, cases are increased or decreased and the algorithm trains via multiple episodes which are the same as epochs in deep learning. The model is then saved subsequently.
    - The file_name is the name of your CSV file, the window_size is the amount of data/steps for the agent to gain its experience from, and the episode_count are the amount of episodes to train the agent. An episode terminates once the terminal state is reached. A good model should be trained for at least 1000-3000 episodes or until there is no longer any improvement. Here, I used window_size=7 to reflect a week.
- eval.py - Once the model has been trained depending on new data, you will be able to test the model for the current policy saved; you can accordingly evaluate the credibility of the model.

The Colab can also be found here (may not be fully updated): https://colab.research.google.com/drive/1cwLCqRVvdGpX41jlN7RjwVWJYMYND1xk?usp=sharing.


