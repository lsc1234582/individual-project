import argparse
import gym
import itertools
import numpy as np
import sys
import tensorflow as tf
import collections

import sklearn.preprocessing

if "../../../" not in sys.path:
  sys.path.append("../../../")
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [2], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.is_training = tf.placeholder(tf.bool, name="is_training")
            hidden_l1 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=96,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                scope="hidden_l1")

            hidden_l1_bn = tf.contrib.layers.batch_norm(
                inputs=hidden_l1,
                is_training = self.is_training,
                scope="hidden_l1_bn")

            # This is just linear classifier
            self.mu = tf.contrib.layers.fully_connected(
                inputs=hidden_l1_bn,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=hidden_l1_bn,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        #print("State")
        #print(state)
        #state = featurize_state(state)
        #print("Featurised state dim")
        #print(state.shape)
        return sess.run(self.action, { self.state: state, self.is_training: False })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        #state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target, self.action: action, self.is_training: True }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [2], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.is_training = tf.placeholder(tf.bool, name="is_training")
            hidden_l1 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=96,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                scope="hidden_l1")

            hidden_l1_bn = tf.contrib.layers.batch_norm(
                inputs=hidden_l1,
                is_training = self.is_training,
                scope="hidden_l1_bn")

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=hidden_l1_bn,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        #state = featurize_state(state)
        return sess.run(self.value_estimate, { self.state: state, self.is_training: False })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        #state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target, self.is_training: True }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
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
    policy_estimator_losses = []
    value_estimator_losses = []

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []

        # One step in the environment
        for t in itertools.count():

            env.render()

            # Take a step
            action = estimator_policy.predict(state)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            ve_loss = estimator_value.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            pe_loss = estimator_policy.update(state, td_error, action)
            policy_estimator_losses.append(pe_loss)
            value_estimator_losses.append(ve_loss)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break

            state = next_state

    return stats, policy_estimator_losses, value_estimator_losses

if __name__ == "__main__":
    #arg_parser = argparse.ArgumentParser(description="Experiment Runner.")
    #arg_parser.add_argument("--experiment-log-path", "-l", dest="experiment_log_path", help="The path to the the log folder.")
    #arg_parser.add_argument("--runner-log-level", dest="log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    #args = arg_parser.parse_args()

    #if args.monitor_log_path and not os.path.exists(args.experiment_log_path):
    #    tf.logging.warning("Experiment log path {} does not exist. Creating a new one.".format(args.experiment_log_path))
    #    os.makedirs(args.experiment_log_path)

    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator(learning_rate=0.001)
    value_estimator = ValueEstimator(learning_rate=0.1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # Note, due to randomness in the policy the number of episodes you need varies
        # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
        stats, pe_losses, ve_losses = actor_critic(env, policy_estimator, value_estimator, 500, discount_factor=0.95)
    np.savetxt(os.path.join("Logs", "RewardsVSEpisodes.csv"), stats.episode_rewards, delimiter=',')
    np.savetxt(os.path.join("Logs", "AverageRewards.log"), np.mean(stats.episode_rewards), delimiter=',')
    np.savetxt(os.path.join("Logs", "PolicyEstimatorTrainingLossVSSteps.csv"), np.array(pe_losses), delimiter=',')
    np.savetxt(os.path.join("Logs", "ValueEstimatorTrainingLossVSSteps.csv"), np.array(ve_losses), delimiter=',')
