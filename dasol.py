import numpy as np
import gym
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS
sigma = 0.01
th_track_angle = -2.0
th_abs_track_angle = 0.2
th_speed = 0.0
alpha = 1e-10
gamma = 0.8

def saturation(value, min, max):
    if value > max:
        return max
    elif value < min:
        return min
    else:
        return value


def get_track_angle(observ_gray):
    angle_append = []
    lat = range(47 - 20, 47 + 20)  # +- 20 is better than +- 10
    lon = range(30, 65)
    for i in lat:
        for j in lon:
            if (observ_gray[j, i] < 120.0):  # in Track
                x = 73.0 - j
                y = 47.0 - i
                angle = np.arctan(y / x) / 3.14 * 180.0
                angle_append.append(angle)
    track_angle = np.mean(angle_append)
    return track_angle, np.abs(track_angle)


def get_speed(observ):
    speed = np.count_nonzero(observ[84:, 13] == [255, 255, 255]) / 3.0
    return speed


def to_gray_scale(observ):
    observ_gray = 0.2126 * observ[:, :, 0] + 0.7152 * observ[:, :, 1] + 0.0722 * observ[:, :, 2]
    return observ_gray


def gaussian_policy(sigma, track_angle, abs_track_angle, th_track_angle, th_abs_track_angle):
    steer = np.random.normal(track_angle * th_track_angle, sigma)
    brake = np.random.normal(abs_track_angle * th_abs_track_angle, sigma)
    return steer, brake





# ---------------------------------------------------------------------------------------------------------------------

env = gym.make('CarRacing-v0')
for i in range(1000):
    observ = env.reset()
    np.set_printoptions(threshold='nan')

    # Initial Action
    action = [0.0, 0.0, 0.0]


    # Variables for Logging
    action_append = []
    score_append = []
    score2_append = []
    rev_return_append = []
    #
    reward_sum = 0
    step_count = 0

    # Start!
    done = False
    while not done:
        # Step fn input; steer, gas, brake
        observ, reward, done, _ = env.step(action)
        env.render()
        reward_sum += reward*(gamma**step_count)


        # Get Features
        speed = get_speed(observ)
        observ_gray = to_gray_scale(observ)
        track_angle, abs_track_angle = get_track_angle(observ_gray)


        # Generate Action
        action_steer, action_brake = gaussian_policy(sigma, track_angle, abs_track_angle, th_track_angle, th_abs_track_angle)
        # action = [track_angle*(-0.030),  0.17,  np.abs(track_angle)*0.013] # TEST 2 : final score > 800 OPTIMAL


        # Make Action Set
        action = [action_steer, 0.17, np.abs(track_angle) * 0.013]
        action_append.append(action)


        # Calculate Score & Log
        temp_score = (action_steer - track_angle * th_track_angle)*track_angle/(sigma**2)
        temp_score2 = (action_brake - abs_track_angle * th_abs_track_angle)*track_angle/(sigma**2)

        score_append.append(temp_score)
        score2_append.append(temp_score2)
        rev_return_append.append(reward_sum)






        # FOR PLOTTING ACTIONS FOR SINGLE EPISODE
        if done == True or step_count > 998:
            temp_return_append = np.array(rev_return_append)
            return_append = temp_return_append[::-1]

            # Parameter Update
            for i in range(return_append.shape[0]):
                th_track_angle += alpha * score_append[i] * return_append[i]
                th_abs_track_angle += alpha * score2_append[i] * return_append[i]
            print(th_track_angle)
            print(th_abs_track_angle)
            print('updated')
            break






            # PLOTTING CODE
            plt.plot(return_append)
            plt.show()


        # FOR PLOTTING STATES
        # plt.imshow(observ_gray, cmap='gray')
        # plt.show()
        # print(observ_gray.shape)
        # print(type(observ_gray))