# Project: Crawler

## Project Details

In this project, we will work with the Crawler environment.

![img](src/img/crawler.gif)


In this environment, there is a creature with 4 arms and 4 four arms.

Goal: The agents must move its body toward the goal direction without falling. 

- CrawlerStaticTarget - Goal direction is always forward.
- CrawlerDynamicTarget- Goal direction is randomized.

Agent Reward Function (independent):

- 0.03 times body velocity in the goal direction.
- 0.01 times body direction alignment with goal direction.

The Observation space consists of 117 variables corresponding to position, rotation, velocity, and angular velocities of each limb plus the acceleration and angular acceleration of the body. Vector Action space: (Continuous) Size of 20, corresponding to target rotations for joints.

The version of environment in this project contains 12 identical agents, each with its own copy of the environment.

## Getting Started

### Unity Environment

For this project, we can download it from one of the links below. You need only select the environment that matches the operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `Crawler_using_PPO/data/` folder, and unzip (or decompress) the file.

__This repo is built in Ubuntu, please change the environment file if your OS is different.__

### Required Python Packages

To install required packages, run `pip install -r src/requirements.txt` in terminal.

### Train the agent

To test the existing agent, please run `python test.py`

To train your own agent, please run `python train.py`


