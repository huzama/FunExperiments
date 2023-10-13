import gym

env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()
env.render()


print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
