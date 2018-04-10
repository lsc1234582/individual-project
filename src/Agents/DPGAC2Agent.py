import numpy as np
import tensorflow as tf

from Utils import ReplayBuffer

class DPGAC2Agent(object):

    def __init__(self, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer, replay_buffer_size=10000):
        self._policy_estimator = policy_estimator
        self._value_estimator = value_estimator
        self._discount_factor = discount_factor
        self._num_episodes = num_episodes
        self._max_episode_length = max_episode_length
        self._minibatch_size = minibatch_size
        self._actor_noise = actor_noise
        self._replay_buffer_size = replay_buffer_size

        # The number of episodes to average total reward over; used for score
        self._num_rewards_to_average = min(50, self._num_episodes - 1)

        self._step = 0
        self._total_reward = 0
        self._average_max_q = 0
        self._rewards_list = []
        self._replay_buffer = ReplayBuffer(self._replay_buffer_size)

        self._current_state = None
        self._action = None
        # Our objective metric
        self._best_average = None

        # To implement early stop
        self._stop_training = False

        # Summary about training
        self._summary_writer = summary_writer

    def score(self):
        return self._best_average

    def act(self, observation, reward, termination, episode_num, is_learning=False):
        #tf.logging.debug("REPLAY_BUFFER")
        #tf.logging.debug(self._replay_buffer)
        self._total_reward += reward
        self._step += 1

        # Putting a limit on how long the a single trial can run for simplicity
        if self._step >= self._max_episode_length:
            termination = True

        if termination:
            # Record cumulative reward of trial
            self._rewards_list.append(self._total_reward)
            average = np.mean(self._rewards_list[-self._num_rewards_to_average:])
            if (len(self._rewards_list) >= self._num_rewards_to_average
                    and (self._best_average is None or self._best_average < average)
                ):
                self._best_average = average

            tf.logging.info("Episode {}/{}, Episode reward {}, Average reward {}, Average Max Q {}".format(episode_num,
                self._num_episodes, self._total_reward, average, self._average_max_q))

            self._summary_writer.writeSummary({"TotalReward": self._total_reward[0]}, episode_num)

            self._summary_writer.writeSummary({"AverageMaxQ": self._average_max_q}, episode_num)
            self._total_reward = 0.0
            self._average_max_q = 0.0
            self._step = 0

        current_state = observation.reshape(1, -1)

        # Initialize the last state and action
        if self._current_state is None:
            self._current_state = current_state
            best_action = self._policy_estimator.predict(self._current_state) + self._actor_noise()
            self._action = best_action
            return best_action, termination

        # Store the current step
        self._replay_buffer.add(self._current_state.squeeze().copy(), self._action.squeeze().copy(), reward, termination,
                current_state.squeeze().copy())

        self._current_state = current_state.copy()

        if is_learning and self._replay_buffer.size() >= self._minibatch_size and not self._stop_training:
            self._train(episode_num)

        best_action = self._policy_estimator.predict(self._current_state)

        self._action = best_action

        return best_action, termination

    def _train(self, episode_num):
        current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch =\
                self._replay_buffer.sample_batch(self._minibatch_size)
        current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
        action_batch = action_batch.reshape(self._minibatch_size, -1)
        reward_batch = reward_batch.reshape(self._minibatch_size, -1)
        next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
        # Calculate targets
        target_q = self._value_estimator.predict_target(
            next_state_batch, self._policy_estimator.predict_target(next_state_batch))

        y_i = []
        for k in range(self._minibatch_size):
            if termination_batch[k]:
                y_i.append(reward_batch[k])
            else:
                y_i.append(reward_batch[k] + self._discount_factor * target_q[k])

        # Update the critic given the targets
        predicted_q_value, _, ve_loss = self._value_estimator.update(
            current_state_batch, action_batch, np.reshape(y_i, (self._minibatch_size, 1)))

        self._average_max_q += np.amax(predicted_q_value)

        # Update the actor policy using the sampled gradient
        a_outs = self._policy_estimator.predict(current_state_batch)
        grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
        self._policy_estimator.update(current_state_batch, grads[0])

        # Update target networks
        self._policy_estimator.update_target_network()
        self._value_estimator.update_target_network()

        # Early stop
        if np.isnan(ve_loss):
            tf.logging.error("Training: value estimator loss too big/nan, stop training")
            self._stop_training = True

        # Some basic summary of training loss
        if self._step % 100 == 0:
            self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)
