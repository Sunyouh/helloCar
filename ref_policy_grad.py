import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
import cv2


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters", [3, 3])
        state = tf.placeholder("float", [None, 3])
        actions = tf.placeholder("float", [None, 3])
        advantages = tf.placeholder("float", [None, 1])
        linear = tf.matmul(state, params)
        # probabilities = tf.nn.softmax(linear)
        # good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), reduction_indices=[1])
        # eligibility = tf.log(good_probabilities) * advantages
        # loss = -tf.reduce_sum(eligibility)
        # optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        # return probabilities, state, actions, advantages, optimizer
        good_probabilities = tf.reduce_sum(tf.mul(linear, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return linear, state, actions, advantages, optimizer


def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None, 3])
        newvals = tf.placeholder("float", [None, 1])
        w1 = tf.get_variable("w1", [3, 10])
        b1 = tf.get_variable("b1", [10])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [10, 3])
        b2 = tf.get_variable("b2", [3])
        calculated = tf.matmul(h1, w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return calculated, state, newvals, optimizer, loss


def run_episode(env, policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    action = env.action_space.sample()

    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    for _ in xrange(200):
        observ_gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        speed = np.count_nonzero(observ_gray[84:, 13] == 255)
        gyro = (np.count_nonzero(observ_gray[84:, 73:] == 76) - np.count_nonzero(observ_gray[84:, 50:72] == 76))/5.0

        # Get TrackAngle
        angle_append = []
        lat = range(47 - 10, 47 + 10)
        lon = range(30, 65)
        for i in lat:
            for j in lon:
                if observ_gray[j, i] < 120.0:  # in Track
                    x = 73.0 - j
                    y = 47.0 - i
                    angle = np.arctan(y / x) / 3.14 * 180.0
                    angle_append.append(angle)
        track_angle = np.mean(angle_append)

        observ_cvt = [track_angle, speed, gyro]

        # calculate policy
        obs_vector = np.expand_dims(observ_cvt, axis=0)
        probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
        # action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # print(probs[0])
        # action = env.action_space.sample()

        steer = probs[0, 0]
        if steer > 1:
            steer = 1
        elif steer < -1:
            steer = -1

        gas = probs[0, 1]
        if gas > 1:
            gas = 1
        elif gas < 0:
            gas = 0

        brake = probs[0, 2]
        if brake > 1:
            brake = 1
        elif brake < 0:
            brake = 0

        action = [steer, gas, brake]
        # print(action)

        # record the transition
        states.append(observ_cvt)
        # actionblank = np.zeros(2)
        # actionblank[action] = 1
        actions.append(action)
        # take the action in the environment
        old_observation = observ_cvt
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward


env = gym.make('CarRacing-v0')
# gym.wrappers.Monitor(env, 'CarRacing-v0/', force=True)
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
reward_batch = 0
for i in xrange(2000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    print(reward)
    # reward_batch += reward
    # if i % 10 == 0:
    #     print(reward_batch/10)
    #     reward_batch = 0
    # if reward == 200:
    #     print "reward 200"
    #     print i
    #     break
t = 0
for _ in xrange(1000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
print t / 1000
# env.monitor.close()