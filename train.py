from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
from datetime import datetime
def MSG(txt):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), str(txt))
from ppo_agent import Agent
import matplotlib.pyplot as plt
import sys
import random

# ## Create Unity environment

# In[2]:

def ppo(env, agent, n_episodes=1700, max_t=1500, print_every=100, n_agent = 12):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): number of episodes to print result
        n_agent (int): number of identical agents in environment
    """
    MSG('start!')
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=print_every)
    t_mean = deque(maxlen=print_every//100)
    scores = []
    best_score = 0.
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent_scores = np.zeros(n_agent)
        for t in range(max_t):
            actions, log_probs, _, values = agent.act(states)
            # get needed information from environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = np.array([1 if t else 0 for t in env_info.local_done])
            agent.save_step([states, values.detach(), actions, log_probs.detach(), rewards, 1 - dones])
            states = next_states
            agent_scores += rewards
            if all(dones) or (t+1) == max_t:
                t_mean.append(t+1)
            if (t+1) == max_t:
                agent.step(next_states)
                break
        score = np.mean(agent_scores)
        scores_deque.append(score)
        scores.append(score)

        if score > best_score:
            torch.save(agent.actor_critic.state_dict(), 'checkpoint.pth')
            best_score = score

        if i_episode % (print_every) == 0:
            print('\rEpisode {}\tCurrent Episode Average Score: {:.2f}\tAverage Score on 100 Episode: {:.2f}\tAverage Step: {:.0f}'.format(i_episode, 
                        score, np.mean(scores_deque), np.mean(t_mean)))

    MSG('\nend!')
    return scores

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
    # train
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, n_agent=n_agent)
    scores = ppo(env, agent, n_agent=n_agent)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig('score.png')
    plt.close(fig)


if __name__ == "__main__":
    main()
