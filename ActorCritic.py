import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
from common import generateRandomVel

import sklearn.preprocessing

from lib import plotting

import os
import random
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import vrep
from sklearn.externals import joblib

from VREPEnvironments import VREPPushTaskEnvironment

if "../" not in sys.path:
  sys.path.append("../")

from collections import deque

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, current_state, action, reward, termination, next_state):
        experience = [current_state, action, reward, termination, next_state]
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        current_state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        termination_batch = np.array([_[3] for _ in batch])
        next_state_batch = np.array([_[4] for _ in batch])

        return current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, data_preprocessor, min_vel, max_vel, learning_rate=0.01, scope="policy_estimator"):
        self.data_preprocessor = data_preprocessor
        self.min_vel = min_vel
        self.max_vel = max_vel
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, 24], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.is_training = tf.placeholder(tf.bool, name="is_training")
            hidden_l1 = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=96,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                scope="hidden_l1")

            hidden_l1_bn = tf.contrib.layers.batch_norm(
                inputs=hidden_l1,
                is_training = self.is_training,
                scope="hidden_l1_bn")

            self.mu = tf.contrib.layers.fully_connected(
                inputs=hidden_l1_bn,
                #inputs=tf.expand_dims(self.state, 0),
                num_outputs=6,
                activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                scope="mu")

            #self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=hidden_l1_bn,
                #inputs=tf.expand_dims(self.state, 0),
                num_outputs=6,
                activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                scope="sigma")

            #self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.MultivariateNormalDiag(self.mu, self.sigma)
            self.action = self.normal_dist.sample()
            self.action = tf.clip_by_value(self.action, self.min_vel, self.max_vel)

            # Loss and train op
            self.loss = -tf.reduce_mean(self.normal_dist.log_prob(self.action) * self.target)
            # Add cross entropy cost to encourage exploration
            # TODO: Shape?
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = self.data_preprocessor.transform(state)
        #state = state.flatten()
        #print("State")
        #print(state)
        return sess.run(self.action, { self.state: state, self.is_training: False})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = self.data_preprocessor.transform(state)
        #state = state.flatten()
        feed_dict = { self.state: state, self.target: target, self.action: action , self.is_training: True }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, data_preprocessor, learning_rate=0.1, scope="value_estimator"):
        self.data_preprocessor = data_preprocessor
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, 24], "state")
            self.target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target")

            # This is just linear classifier
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            hidden_l1 = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=96,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                scope="hidden_l1")

            hidden_l1_bn = tf.contrib.layers.batch_norm(
                inputs=hidden_l1,
                is_training = self.is_training,
                scope="hidden_l1_bn")

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=hidden_l1_bn,
                num_outputs=1,
                activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                scope="output_layer")
            #self.value_estimate = tf.squeeze(self.output_layer)
            self.value_estimate = self.output_layer

            self.loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.target))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = self.data_preprocessor.transform(state)
        #state = state.flatten()
        return sess.run(self.value_estimate, { self.state: state, self.is_training: False})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = self.data_preprocessor.transform(state)
        #state = state.flatten()
        feed_dict = { self.state: state, self.target: target, self.is_training: True}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, episode_length, sess, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["current_state", "action", "reward", "next_state"])

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    current_state = env.reset(True)
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action

        episode = []

        # One step in the environment
        for t in range(episode_length):

            # Take a step
            action = estimator_policy.predict(current_state.reshape(1, -1), sess)
            next_state, reward = env.step(action)
            current_state = current_state.flatten()
            action = action.flatten()
            reward = reward.flatten()
            next_state = next_state.flatten()

            # Terminate if the current velocity is too high (avoid bad data)
            #if np.any(np.abs(next_state[:6]) >= env.MAX_JOINT_VELOCITY ):
            #    break
            # Keep track of the transition
            episode.append(Transition(
              current_state=current_state, action=action, reward=reward[0], next_state=next_state))
            replay_buffer.add(current_state, action, reward, None, next_state)

            # Update statistics
            stats.episode_rewards[i_episode] += reward[0]
            stats.episode_lengths[i_episode] = t

            # Update policy and value estimators when replay buffer is full
            if replay_buffer.size() >= POLICY_BATCH_SIZE:
                current_state_batch, action_batch, reward_batch, _, next_state_batch =\
                        replay_buffer.sample_batch(POLICY_BATCH_SIZE)
                # Calculate TD Target
                #print(current_state_batch.shape)
                #print(action_batch.shape)
                #print(reward_batch.shape)
                #print(next_state_batch.shape)
                value_next = estimator_value.predict(next_state_batch, sess)
                td_target = reward_batch + discount_factor * value_next
                td_error = td_target - estimator_value.predict(current_state_batch)
                #print("REWARD")
                #print(reward)
                #print("ACTION")
                #print(action)
                ##_ = input()

                ## Update the value estimator
                estimator_value.update(current_state_batch, td_target, sess)

                # Update the policy estimator
                # using the td error as our advantage estimate
                estimator_policy.update(current_state_batch, td_error, action_batch, sess)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")


            current_state = next_state

        current_state = env.reset(False)

    return stats


def getAdvantages(rewards, discount_factor):
    eps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running_discounted_advantages = 0
    for i in range(eps - 1, -1, -1):
        running_discounted_advantages = running_discounted_advantages * discount_factor + rewards[i]
        advantages[i] = running_discounted_advantages
    return advantages


def getGreedyFactor(eps):
    """
    Greedy factor scheduler
    """
    return 0.5 -  0.5 / (1 + np.exp(-0.01 * (eps - 900)))

def getActionCov(eps, max_vel, decay_rate, mid_point):
    """
    Action cov scheduler
    """
    return max_vel - max_vel / (1 + np.exp(-decay_rate * (eps - mid_point)))


NUM_EPS = 100
EPS_LENGTH = 200
DISCOUNT_FACTOR = 0.95

REPLAY_BUFFER_SIZE = 5000
POLICY_BATCH_SIZE = 128
POLICY_LEARNING_RATE = 0.001

VALUE_LEARNING_RATE = 0.05


if __name__ == "__main__":
    # Preprocessing: Normalize to zero mean and unit variance
    #state_samples = []
    #with VREPPushTaskEnvironment() as env:
    #    current_state = env.reset(is_first_reset=True)
    #    state_samples = [current_state]
    #    for ep in range(100):
    #        for step in range(600):
    #            action = generateRandomVel(env.MAX_JOINT_VELOCITY_DELTA)
    #            action = action.reshape((1, -1))
    #            next_state, reward = env.step(action)
    #            state_samples.append(next_state)
    #        current_state = env.reset(is_first_reset=False)
    #state_samples = np.concatenate(state_samples, axis=0)
    #np.save("VREPPushTaskEnvironmentStateSamples.npy", state_samples)

    #state_samples = np.load("VREPPushTaskEnvironmentStateSamples.npy")
    #data_preprocessor = sklearn.preprocessing.StandardScaler()
    #data_preprocessor.fit(state_samples)
    #joblib.dump(data_preprocessor, "VREPPushTaskEnvironmentStateSamplePreprocessor.pkl")

    data_preprocessor = joblib.load("VREPPushTaskEnvironmentStateSamplePreprocessor.pkl")
    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.Session() as sess, VREPPushTaskEnvironment() as env:
        policy_estimator = PolicyEstimator(data_preprocessor, -1 * env.MAX_JOINT_VELOCITY_DELTA,
                env.MAX_JOINT_VELOCITY_DELTA, learning_rate=POLICY_LEARNING_RATE)
        value_estimator = ValueEstimator(data_preprocessor, learning_rate=VALUE_LEARNING_RATE)
        sess.run(tf.initialize_all_variables())
        # Note, due to randomness in the policy the number of episodes you need varies
        # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
        stats = actor_critic(env, policy_estimator, value_estimator, NUM_EPS, EPS_LENGTH, sess,
                discount_factor=DISCOUNT_FACTOR)
        plotting.plot_episode_stats(stats, smoothing_window=1)
