from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
from datetime import datetime
from ppo_agent import Agent

import sys
import random
from collections import namedtuple, deque

from unityagents import UnityEnvironment

def main():
    env = UnityEnvironment(file_name='data/Crawler_Linux/Crawler.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    n_agent = len(env_info.agents)
    print("there are %d identical agents in the env"%n_agent)
    # size of each action
    action_size = brain.vector_action_space_size
    # size of state space 
    state_size = env_info.vector_observations.shape[1]
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, n_agent=n_agent)
    # load trained model
    agent.actor_critic.load_state_dict(torch.load('model/checkpoint.pth'))
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    agent_scores = np.zeros(n_agent)
    for t in range(1500):
        actions, log_probs, _, values = agent.act(states,scale=0.)
        # get needed information from environment
        env_info = env.step(actions)[brain_name]
        agent_scores += env_info.rewards
        next_states = env_info.vector_observations
        dones = np.array([1 if t else 0 for t in env_info.local_done])
        states = next_states
        if all(dones):
            break
    env.close()
    print(np.mean(agent_scores))
    
if __name__ == "__main__":
    main()
