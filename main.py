import numpy as np
import gym
import cv2

env = gym.make('CarRacing-v0')

observ_samples = []

for i in range(1000):
    observ = env.reset()
    observ_samples.append(observ)

    # print(env.action_space)
    np.set_printoptions(threshold='nan')

    done = False
    while not done:
        # Step fn input; steer, gas, brake

        observ, reward, done, _ = env.step(env.action_space.sample())
        # observ_samples.append(observ)

        env.render()

        # convert observ rgb to grayscale
        cvted = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)

        print(cvted.shape, reward)

        # show grayscale-converted state
        cv2.imshow('img', cvted)
        cv2.waitKey(1)


# observ_samples = np.array(observ_samples)
#
# env = gym.wrappers.Monitor(env, 'monitor-folder', force=True)
