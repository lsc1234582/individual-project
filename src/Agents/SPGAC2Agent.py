import numpy as np
import tensorflow as tf

from Utils.Utils import ReplayBuffer
from Utils.Utils import getModuleLogger

# Module logger
logger = getModuleLogger(__name__)

class SPGAC2Agent(object):

    def __init__(self, sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, summary_writer, replay_buffer_size=10000):
        self._sess = sess
        self._policy_estimator = policy_estimator
        self._value_estimator = value_estimator
        self._discount_factor = discount_factor
        self._num_episodes = num_episodes
        self._max_episode_length = max_episode_length
        self._minibatch_size = minibatch_size
        self._replay_buffer_size = replay_buffer_size

        # The number of episodes to average total reward over; used for score
        self._num_rewards_to_average = min(50, self._num_episodes - 1)

        self._step = 0
        self._total_reward = 0
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
        #logger.debug("REPLAY_BUFFER")
        #logger.debug(self._replay_buffer)
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

            logger.info("Episode {}/{}, Episode reward {}, Average reward {}".format(episode_num,
                self._num_episodes, self._total_reward, average))

            self._summary_writer.writeSummary({"TotalReward": self._total_reward[0]}, episode_num)
            self._total_reward = 0.0
            self._step = 0

        current_state = observation.reshape(1, -1)

        # Initialize the last state and action
        if self._current_state is None:
            self._current_state = current_state
            best_action = self._policy_estimator.predict(self._current_state)
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
        current_state_batch, action_batch, reward_batch, _, next_state_batch =\
                self._replay_buffer.sample_batch(self._minibatch_size)
        current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
        action_batch = action_batch.reshape(self._minibatch_size, -1)
        reward_batch = reward_batch.reshape(self._minibatch_size, -1)
        next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
        # Calculate TD Target
        value_next = self._value_estimator.predict(next_state_batch)
        td_target = reward_batch + self._discount_factor * value_next
        td_error = td_target - self._value_estimator.predict(current_state_batch)

        # Update the value estimator
        ve_loss = self._value_estimator.update(current_state_batch, td_target)
        # Update the policy estimator
        # using the td error as our advantage estimate
        pe_loss = self._policy_estimator.update(current_state_batch, td_error, action_batch)

        # Early stop
        if np.isnan(pe_loss):
            logger.error("Training: policy estimator loss too big/nan, stop training")
            self._stop_training = True

        # Some basic summary of training loss
        if self._step % 100 == 0:
            self._summary_writer.writeSummary({"PolicyEstimatorTrainLoss": pe_loss,
                "ValueEstimatorTrainLoss": ve_loss}, self._step)
