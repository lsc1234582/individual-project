import numpy as np
import tensorflow as tf

from Utils import ReplayBuffer
from Utils import getModuleLogger

# Module logger
logger = getModuleLogger(__name__)

class DPGAC2Agent(object):

    def __init__(self, sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise, summary_writer,
            estimator_dir, estimator_saver, estimator_save_freq):
        self._sess = sess
        self._policy_estimator = policy_estimator
        self._value_estimator = value_estimator
        self._discount_factor = discount_factor
        self._num_episodes = num_episodes
        self._max_episode_length = max_episode_length
        self._minibatch_size = minibatch_size
        self._actor_noise = actor_noise
        self._replay_buffer_size = replay_buffer_size

        # The number of episodes to average total reward over; used for score
        self._num_rewards_to_average = min(100, self._num_episodes - 1)

        self._step = 0
        self._total_reward = 0
        self._max_q = 0
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
        self._estimator_dir = estimator_dir
        self._estimator_saver = estimator_saver
        self._estimator_save_freq = estimator_save_freq

    def save(self, estimator_dir, name="DPGAC2Agent", step=None, write_meta_graph=False):
        self._estimator_saver.save(self._sess, "{}/{}".format(self._estimator_dir, name), global_step=step, write_meta_graph=write_meta_graph)
        logger.info("DPGAC2Agent saved")

    def load(self, estimator_dir, best_estimator_name=None):
        # Load estiamtor
        if best_estimator_name:
            logger.info("Loading best DPGAC2Agent")
            self._estimator_saver.restore(self._sess, "{}/{}".format(self._estimator_dir, best_estimator_name))
        else:
            logger.info("Loading most recent DPGAC2Agent")
            self._estimator_saver.restore(self._sess, tf.train.latest_checkpoint(estimator_dir))
        logger.info("DPGAC2Agent loaded")

    def score(self):
        return self._best_average

    def act(self, observation, reward, termination, episode_start_num, episode_num, episode_num_var, is_learning=False):
        #logger.debug("REPLAY_BUFFER")
        #logger.debug(self._replay_buffer)
        self._total_reward += reward
        self._step += 1

        # Putting a limit on how long the a single trial can run for simplicity
        if self._step > self._max_episode_length:
            termination = True

        if termination:
            # Record cumulative reward of trial
            self._rewards_list.append(self._total_reward)
            average = np.mean(self._rewards_list[-self._num_rewards_to_average:])
            # Update episode number variable
            tf.assign(episode_num_var, episode_num)
            if self._best_average is None or self._best_average < average:
                self._best_average = average
                improve_str = '*'
                # Save best estimator so far
                if is_learning:
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_saver, name="DPGAC2AgentBest", step=episode_num, write_meta_graph=False)
            else:
                improve_str = ''

            #if is_learning and episode_num % self._estimator_save_freq == 0:
            #    logger.info("Saving agent checkpoints")
            #    self.save(self._estimator_dir, step=episode_num, write_meta_graph=False)

            log_string = "Episode {0:>5}/{1:>5} ({2:>5}/{3:>5} in this run), " +\
                         "R {4:>9.3f}, Ave R {5:>9.3f} {7}, Ave Max Q {6:>9.3f}"

            logger.info(log_string.format(episode_num, self._num_episodes + episode_start_num - 1,
                episode_num - episode_start_num + 1,
                self._num_episodes,
                self._total_reward[0], average, self._max_q / float(self._step), improve_str))

            self._summary_writer.writeSummary({
                "TotalReward": self._total_reward[0],
                "AverageMaxQ": self._max_q / float(self._step),
                "BestAverage": self._best_average,
                }, episode_num)

            self._total_reward = 0.0
            self._max_q = 0.0
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
            self._train()

        best_action = self._policy_estimator.predict(self._current_state)

        self._action = best_action

        return best_action, termination

    def _train(self):
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

        self._max_q += np.amax(predicted_q_value)

        # Update the actor policy using the sampled gradient
        a_outs = self._policy_estimator.predict(current_state_batch)
        grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
        self._policy_estimator.update(current_state_batch, grads[0])

        # Update target networks
        self._policy_estimator.update_target_network()
        self._value_estimator.update_target_network()

        # Early stop
        if np.isnan(ve_loss):
            logger.error("Training: value estimator loss is nan, stop training")
            self._stop_training = True

        # Some basic summary of training loss
        #if self._step % 100 == 0:
        #    self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)
