import numpy as np
import gym

env = gym.make('CarRacing-v0')

observ_samples = []

for i in range(1000):
    observ = env.reset()
    observ_samples.append(observ)

    print(env.action_space.n)

    done = False
    while not done:
        action = np.random.randint(0, env.action_space)
        observ, reward, done, _ = env.step(action)
        observ_samples.append(observ)

observ_samples = np.array(observ_samples)

env = gym.wrappers.Monitor(env, 'monitor-folder', force=True)