import numpy as np
import gym
import matplotlib.pyplot as plt


env = gym.make('CarRacing-v0')
for i in range(1000):
    observ = env.reset()


    np.set_printoptions(threshold='nan')

    action = [-0.2, 0.1, 0.0]
    action_append = []
    done = False
    while not done:
        # Step fn input; steer, gas, brake



        observ, reward, done, _ = env.step(action)
        env.render()
        observ_gray = 0.2126 * observ[:, :, 0] + 0.7152 * observ[:, :, 1] + 0.0722 * observ[:, :, 2]
        speed = np.count_nonzero(observ[84:, 13] == [255, 255, 255])/3.0
        print('sp:', speed)


        # Get TrackAngle
        angle_append = []
        lat = range(47-10, 47+10)
        lon = range(30, 65)
        for i in lat:
            for j in lon:
                if (observ_gray[j, i] < 120.0): # in Track
                    x = 73 - j
                    y = 47 - i
                    angle = np.arctan(y/x)/3.14*180.0
                    angle_append.append(angle)
        track_angle = np.mean(angle_append)
        # print(track_angle)


        # Generate Action
        # action = [track_angle*(-0.02), 0.05-np.abs(track_angle)*0.03, 0.0] # TEST 1
        action = [track_angle*(-0.030),  0.17,  np.abs(track_angle)*0.010] # TEST 2 : final score > 800 OPTIMAL
        # action = [track_angle*(-0.030),  0.18,  np.abs(track_angle)*0.013] # TEST 3 : final score > 850
        action = [track_angle*(-0.030),   0.25 - np.abs(track_angle)*(0.10),  np.abs(track_angle)*0.013]

        action_append.append(action)




        # FOR PLOTTING ACTIONS FOR SINGLE EPISODE
        if done == True:
            plt.plot(action_append)
            plt.show()


        # FOR PLOTTING STATES
        plt.imshow(observ_gray, cmap='gray')
        plt.show()
        print(observ_gray.shape)
        print(type(observ_gray))