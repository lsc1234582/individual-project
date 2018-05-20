import numpy as np
import random
import tensorflow as tf
import time
import pprint

from Utils import ReplayBuffer
from Utils import getModuleLogger
from Utils import generateRandomAction

# Module logger
logger = getModuleLogger(__name__)

class AgentBase(object):
    def __init__(self, sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1):
        self._sess = sess
        self._policy_estimator = policy_estimator
        self._value_estimator = value_estimator
        self._discount_factor = discount_factor
        self._num_episodes = num_episodes
        self._max_episode_length = max_episode_length
        self._minibatch_size = minibatch_size
        self._actor_noise = actor_noise

        self._step = 0

        self._last_state = None
        self._last_action = None

        # To implement early stop
        self._stop_training = False
        self._last_average = None
        self._num_non_imp_eps = 0  # Maximal number of non-improving episodes
        self._max_num_non_imp_eps = 200  # Maximal number of non-improving episodes

        # Summary and checkpoints
        # Agent stats related
        self._stats_sample = None
        self._stats_sample_size = 1024

        # The number of episodes to average total reward over; used for score
        self._num_rewards_to_average = min(100, self._num_episodes - 1)

        self._episode_return = 0
        self._episode_returns = []
        # Rollout stats related
        # Our objective metric
        self._best_average_episode_return = None
        self._stats_start_time = time.time()
        self._stats_epoch_episode_returns = []
        self._stats_epoch_episode_steps = []
        self._stats_epoch_actions = []
        self._stats_epoch_Q = []
        self._stats_epoch_critic_loss = []
        self._stats_tot_steps = 0
        self._log_stats_freq = log_stats_freq

        self._replay_buffer = replay_buffer
        self._summary_writer = summary_writer
        self._estimator_save_dir = estimator_save_dir
        self._estimator_saver_recent = estimator_saver_recent
        self._estimator_saver_best = estimator_saver_best
        self._recent_save_freq = recent_save_freq
        self._replay_buffer_save_dir = replay_buffer_save_dir
        self._replay_buffer_save_freq = replay_buffer_save_freq

        self._normalize_states = normalize_states
        self._state_rms = state_rms
        self._normalize_returns = normalize_returns
        self._return_rms = return_rms

        # Number of updates per training step
        self._num_updates = num_updates

    def _sampleBatch(self, batch_size, **kwargs):
        return self._replay_buffer.sample_batch(batch_size=batch_size)

    def getStats(self):
        if self._stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self._stats_sample = self._sampleBatch(self._stats_sample_size)
        # Add normalization stats
        ops = []
        names = []
        if self._normalize_returns:
            ops += [self._return_rms.mean, self._return_rms.std]
            names += ['return_rms_mean', 'return_rms_std']

        # TODO: full state vector instead of reduced value? Can output it in log file instead of tensorboard
        if self._normalize_states:
            ops += [tf.reduce_mean(self._state_rms.mean), tf.reduce_mean(self._state_rms.std)]
            names += ['state_rms_mean', 'state_rms_std']
        values = self._sess.run(ops)
        assert len(names) == len(values)
        stats = dict(zip(names, values))
        stats_sample_q = self._value_estimator.predict(self._stats_sample[0], self._stats_sample[1])
        stats["agent_sample_Q_mean"] = np.mean(stats_sample_q)
        stats["agent_sample_Q_std"] = np.std(stats_sample_q)
        stats_sample_action = self._policy_estimator.predict(self._stats_sample[0])
        stats["agent_sample_action_mean"] = np.mean(stats_sample_action)
        stats["agent_sample_action_std"] = np.std(stats_sample_action)
        stats_sample_action_q = self._value_estimator.predict(self._stats_sample[0], stats_sample_action)
        stats["agent_sample_action_Q_mean"] = np.mean(stats_sample_action_q)
        stats["agent_sample_action_Q_std"] = np.std(stats_sample_action_q)

        return stats

    def _logStats(self, episode_num):
        """
        Logging happens at the end of every logging epoch, which is determined by the logging frequency, in number of
        episodes
        """
        stats_tot_duration = time.time() - self._stats_start_time
        stats = self.getStats()
        combined_stats = stats.copy()
        # Epoch statistics.
        combined_stats['epoch/episode_return'] = np.mean(self._stats_epoch_episode_returns)
        combined_stats['epoch/episode_steps'] = np.mean(self._stats_epoch_episode_steps)
        # TODO: Again, full action vector instead of reduced value?
        combined_stats['epoch/actions_mean'] = np.mean(self._stats_epoch_actions)
        combined_stats['epoch/actions_std'] = np.std(self._stats_epoch_actions)
        combined_stats['epoch/Q_mean'] = np.mean(self._stats_epoch_Q)
        combined_stats['epoch/Q_std'] = np.std(self._stats_epoch_Q)
        #combined_stats['epoch/actor_loss'] = np.mean(self._stats_epoch_actor_loss)
        combined_stats['epoch/critic_loss'] = np.mean(self._stats_epoch_critic_loss)
        # Clear epoch statistics.
        self._stats_epoch_episode_returns = []
        self._stats_epoch_episode_steps = []
        self._stats_epoch_actions = []
        self._stats_epoch_Q = []
        self._stats_epoch_critic_loss = []

        # Evaluation statistics.
        #if eval_env is not None:
        #    combined_stats['epoch/eval/episode_return'] = np.mean(self._epoch_eval_episode_return)
        #    combined_stats['epoch/eval/Q_mean'] = np.mean(self._epoch_eval_Q_mean)
        #    combined_stats['epoch/eval/Q_std'] = np.std(self._epoch_eval_Q_mean)

        # Total statistics.
        combined_stats["total/score"] = self.score()
        combined_stats['total/duration'] = stats_tot_duration
        combined_stats['total/steps_per_second'] = float(self._stats_tot_steps) / float(stats_tot_duration)
        combined_stats['total/steps'] = self._stats_tot_steps


        #pprint.pprint(combined_stats)
        self._summary_writer.writeSummary(combined_stats, episode_num)


    def initialize(self):
        """
        Target networks share the same parameters with the behaviourial networks at the beginning
        """
        logger.info("Initializing agent {}".format(self.__class__.__name__))
        self._sess.run(tf.global_variables_initializer())
        self._policy_estimator.update_target_network(tau=1.0)
        self._value_estimator.update_target_network(tau=1.0)


    def save(self, estimator_save_dir, is_best=False, step=None, write_meta_graph=False):
        if write_meta_graph:
            tf.train.export_meta_graph(filename="{}/{}.meta".format(estimator_save_dir, self.__class__.__name__))
            logger.info("{} meta graph saved".format(self.__class__.__name__))
        else:
            if is_best:
                self._estimator_saver_best.save(self._sess, "{}/best/{}".format(estimator_save_dir, self.__class__.__name__), global_step=step,
                    write_meta_graph=False)
            else:
                self._estimator_saver_recent.save(self._sess, "{}/recent/{}".format(estimator_save_dir, self.__class__.__name__), global_step=step,
                    write_meta_graph=False)
            logger.info("{} saved".format(self.__class__.__name__))

    def saveReplayBuffer(self, path):
        self._replay_buffer.save(path)
        logger.info("Replay buffer saved")

    def load(self, estimator_dir, is_best=False):
        if is_best:
            logger.info("Loading the best {}".format(self.__class__.__name__))
            self._estimator_saver_best.restore(self._sess,
                    tf.train.latest_checkpoint("{}/best".format(estimator_dir)))
        else:
            logger.info("Loading the most recent {}".format(self.__class__.__name__))
            self._estimator_saver_recent.restore(self._sess,
                    tf.train.latest_checkpoint("{}/recent".format(estimator_dir)))
        logger.info("{} loaded".format(self.__class__.__name__))

    def loadReplayBuffer(self, path):
        self._replay_buffer.load(path)
        logger.info("Replay buffer loaded")

    def score(self):
        return self._best_average_episode_return

    def reset(self):
        self._actor_noise.reset()

    def act(self, observation, last_reward, termination, episode_start_num, episode_num, episode_num_var, is_learning=False):
        self._episode_return += last_reward
        self._step += 1
        self._stats_tot_steps += 1
        # Putting a limit on how long the a single trial can run for simplicity
        if self._step > self._max_episode_length:
            termination = True

        current_state = observation.reshape(1, -1)

        # Initialize the last state and action
        if self._last_state is None:
            self._last_state = current_state
            best_action = self._policy_estimator.predict(self._last_state)
            # For stats purpose
            best_action_q = self._value_estimator.predict(self._last_state, best_action)
            self._stats_epoch_actions.append(best_action)
            self._stats_epoch_Q.append(best_action_q)
            # Add exploration noise when training
            if is_learning:
                best_action += self._actor_noise()
            self._last_action = best_action
            return best_action, termination

        # Store the last step
        self._replay_buffer.add(self._last_state.squeeze().copy(), self._last_action.squeeze().copy(), last_reward,
                current_state.squeeze().copy(), termination)

        if self._normalize_states:
            self._state_rms.update(np.array([self._last_state]))

        self._last_state = current_state.copy()

        if is_learning and self._replay_buffer.size() >= self._minibatch_size and not self._stop_training:
            self._train()

        best_action = self._policy_estimator.predict(self._last_state)
        # For stats purpose
        best_action_q = self._value_estimator.predict(self._last_state, best_action)
        self._stats_epoch_actions.append(best_action)
        self._stats_epoch_Q.append(best_action_q)
        # Add exploration noise when training
        if is_learning:
            best_action += self._actor_noise()

        self._last_action = best_action

        if termination:
            # Record cumulative reward of trial
            self._episode_returns.append(self._episode_return)
            self._stats_epoch_episode_returns.append(self._episode_return)
            self._stats_epoch_episode_steps.append(self._step)
            average = np.mean(self._episode_returns[-self._num_rewards_to_average:])

            # Update episode number variable
            self._sess.run(tf.assign(episode_num_var, episode_num))

            # Check for improvements
            if len(self._episode_returns) >= self._num_rewards_to_average and\
                (self._best_average_episode_return is None or self._best_average_episode_return < average):
                self._best_average_episode_return = average
                improve_str = '*'
            else:
                improve_str = ''

            # Log the episode summary
            log_string = "Episode {0:>5}/{1:>5} ({2:>5}/{3:>5} in this run), " +\
                         "R {4:>9.3f}, Ave R {5:>9.3f} {6}"

            logger.info(log_string.format(episode_num, self._num_episodes + episode_start_num - 1,
                episode_num - episode_start_num + 1,
                self._num_episodes,
                self._episode_return[0], average, improve_str))

            # Log stats
            if is_learning and episode_num % self._log_stats_freq == 0:
                self._logStats(episode_num)

            # Checkpoint
            if is_learning:
                if improve_str == '*':
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_save_dir, is_best=True, step=episode_num, write_meta_graph=False)
                if (episode_num % self._recent_save_freq == 0 or episode_num >= self._num_episodes +\
                    episode_start_num - 1):
                    logger.info("Saving agent checkpoints")
                    self.save(self._estimator_save_dir, step=episode_num, write_meta_graph=False)

            # Save replay buffer
            if (not self._replay_buffer_save_dir is None) and \
                (episode_num % self._replay_buffer_save_freq == 0 or episode_num >= self._num_episodes +\
                episode_start_num - 1):
                logger.info("Saving replay buffer")
                self.saveReplayBuffer(self._replay_buffer_save_dir)

            # Check for convergence
            if self._last_average and average <= self._last_average:
                self._num_non_imp_eps += 1
            else:
                self._num_non_imp_eps = 0
            if self._num_non_imp_eps >= self._max_num_non_imp_eps:
                logger.info("Agent is not improving; stop training")
                self._stop_training = True
            self._last_average = average

            # Reset for new episode
            self._episode_return = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None
            self.reset()

        return best_action, termination

class DPGAC2Agent(AgentBase):
    def __init__(self, sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1):
         super().__init__(sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates,
            log_stats_freq)

    def _train(self):
        for _ in range(self._num_updates):
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch =\
                    self._sampleBatch(self._minibatch_size)
            current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
            # Calculate targets
            predicted_target_q = self._value_estimator.predict_target(
                next_state_batch, self._policy_estimator.predict_target(next_state_batch))

            td_target = []
            for k in range(self._minibatch_size):
                if termination_batch[k]:
                    td_target.append(reward_batch[k])
                else:
                    td_target.append(reward_batch[k] + self._discount_factor * predicted_target_q[k])

            # Update the critic given the targets
            _, ve_loss = self._value_estimator.update(
                current_state_batch, action_batch, np.reshape(td_target, (self._minibatch_size, 1)))
            self._stats_epoch_critic_loss.append(ve_loss)

            # NB: Use td_target because it's not pure estimate (reward as samples)
            if self._normalize_returns:
                self._return_rms.update(td_target.flatten())

            # Update the actor policy using the sampled gradient
            a_outs = self._policy_estimator.predict(current_state_batch)
            grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
            self._policy_estimator.update(current_state_batch, grads[0])

            # Early stop
            if np.isnan(ve_loss):
                logger.error("Training: value estimator loss is nan, stop training")
                self._stop_training = True

            # Some basic summary of training loss
            #if self._step % 100 == 0:
            #    self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)
        # Update target networks
        self._policy_estimator.update_target_network()
        self._value_estimator.update_target_network()


class DPGAC2WithDemoAgent(AgentBase):
    def __init__(self, sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise,
            summary_writer, imitation_summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1):
         super().__init__(sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates)
         self._imitation_summary_writer = imitation_summary_writer


    def _trainPolicyWithDemo(self, epochs):
        """
        Train a policy nn with imitation learning on all expriences from replay buffer

        Note that if there's only one replay buffer in use,make sure this method is not used to imitate 'bad' experiences.

        Parameters
        ----------
        epochs:  int - Number of iterations to train nns on all experiences
        """
        logger.info("Training policy with demo")
        for i in range(epochs):
            for exp in self._sampleBatch(self._minibatch_size):
                current_state_batch, action_batch, _, _, _ = exp
                current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
                action_batch = action_batch.reshape(self._minibatch_size, -1)
                imitation_loss = self._policy_estimator.updateImitation(current_state_batch, action_batch)
                self._imitation_summary_writer.writeSummary({
                    "TrainLoss": imitation_loss
                    }, i)

    def _trainPolicyCriticFull(self, epochs):
        """
        Train a policy nn and critic nn on all experiences from replay buffer sampled in minibatches

        Parameters
        ----------
        epochs:  int - Number of iterations to train nns on all experiences
        """
        for i in range(epochs):
            for exp in self._sampleBatch(self._minibatch_size):
                current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch = exp
                current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
                action_batch = action_batch.reshape(self._minibatch_size, -1)
                reward_batch = reward_batch.reshape(self._minibatch_size, -1)
                next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
                # Calculate targets
                predicted_target_q = self._value_estimator.predict_target(
                    next_state_batch, self._policy_estimator.predict_target(next_state_batch))

                td_target = []
                for k in range(self._minibatch_size):
                    if termination_batch[k]:
                        td_target.append(reward_batch[k])
                    else:
                        td_target.append(reward_batch[k] + self._discount_factor * predicted_target_q[k])

                # Update the critic given the targets
                predicted_q, _, ve_loss = self._value_estimator.update(
                    current_state_batch, action_batch, np.reshape(td_target, (self._minibatch_size, 1)))

                # Update the actor policy using the sampled gradient
                a_outs = self._policy_estimator.predict(current_state_batch)
                grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
                self._policy_estimator.update(current_state_batch, grads[0])

                # Early stop
                if np.isnan(ve_loss):
                    logger.error("Training: value estimator loss is nan, stop training")
                    self._stop_training = True

                # Some basic summary of training loss
                #if self._step % 100 == 0:
                #    self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)

            if i % 10 == 0:
                # Update target networks
                self._policy_estimator.update_target_network()
                self._value_estimator.update_target_network()


    def act(self, observation, last_reward, termination, episode_start_num, episode_num, episode_num_var, is_learning=False):
        self._episode_return += last_reward
        self._step += 1
        # Putting a limit on how long the a single trial can run for simplicity
        if self._step > self._max_episode_length:
            termination = True

        # Initial step at Initial episode
        if episode_num == episode_start_num and self._last_state is None:
            self._trainPolicyWithDemo(200)

        current_state = observation.reshape(1, -1)
        # Initial step of an episode
        if self._last_state is None:
            # TODO: Train policy full
            #self._trainPolicyCriticFull(1)
            # Initialize the last state and action
            self._last_state = current_state
            best_action = self._policy_estimator.predict(self._last_state)
            # Add exploration noise when training
            if is_learning:
                best_action += self._actor_noise()
            self._last_action = best_action
            return best_action, termination

        # Store the last step
        self._replay_buffer.add(self._last_state.squeeze().copy(), self._last_action.squeeze().copy(), last_reward,
                current_state.squeeze().copy(), termination)
        if self._normalize_states:
            self._state_rms.update(np.array([self._last_state]))


        self._last_state = current_state.copy()

        if is_learning and self._replay_buffer.size() >= self._minibatch_size and not self._stop_training:
            self._train()

        best_action = self._policy_estimator.predict(self._last_state)
        # Add exploration noise when training
        if is_learning:
            best_action += self._actor_noise()

        self._last_action = best_action

        if termination:
            # Record cumulative reward of trial
            self._episode_returns.append(self._episode_return)
            average = np.mean(self._episode_returns[-self._num_rewards_to_average:])

            # Update episode number variable
            self._sess.run(tf.assign(episode_num_var, episode_num))

            # Check for improvements
            if len(self._episode_returns) >= self._num_rewards_to_average and\
                (self._best_average_episode_return is None or self._best_average_episode_return < average):
                self._best_average_episode_return = average
                improve_str = '*'
            else:
                improve_str = ''

            # Log the episode summary
            log_string = "Episode {0:>5}/{1:>5} ({2:>5}/{3:>5} in this run), " +\
                         "R {4:>9.3f}, Ave R {5:>9.3f} {7}, Ave Max Q {6:>9.3f}"

            logger.info(log_string.format(episode_num, self._num_episodes + episode_start_num - 1,
                episode_num - episode_start_num + 1,
                self._num_episodes,
                self._episode_return[0], average, self._max_q / float(self._step), improve_str))

            # Checkpoint
            if is_learning:
                if improve_str == '*':
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_save_dir, is_best=True, step=episode_num, write_meta_graph=False)
                if (episode_num % self._recent_save_freq == 0 or episode_num >= self._num_episodes +\
                    episode_start_num - 1):
                    logger.info("Saving agent checkpoints")
                    self.save(self._estimator_save_dir, step=episode_num, write_meta_graph=False)
                # Save summary
                self._summary_writer.writeSummary({
                    "TotalReward": self._episode_return[0],
                    "AverageMaxQ": self._max_q / float(self._step),
                    "BestAverage": self._best_average_episode_return,
                    }, episode_num)
            # Save replay buffer
            if (not self._replay_buffer_save_dir is None) and \
                (episode_num % self._replay_buffer_save_freq == 0 or episode_num >= self._num_episodes +\
                episode_start_num - 1):
                logger.info("Saving replay buffer")
                self.saveReplayBuffer(self._replay_buffer_save_dir)

            # Check for convergence
            if self._last_average and average <= self._last_average:
                self._num_non_imp_eps += 1
            else:
                self._num_non_imp_eps = 0
            if self._num_non_imp_eps >= self._max_num_non_imp_eps:
                logger.info("Agent not improving; stop training")
                self._stop_training = True
            self._last_average = average

            # Reset for new episode
            self._episode_return = 0.0
            self._max_q = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None

        return best_action, termination

    def _train(self):
        for _ in range(self._num_updates):
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch =\
                    self._sampleBatch(self._minibatch_size)
            current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
            # Calculate targets
            predicted_target_q = self._value_estimator.predict_target(
                next_state_batch, self._policy_estimator.predict_target(next_state_batch))

            td_target = []
            for k in range(self._minibatch_size):
                if termination_batch[k]:
                    td_target.append(reward_batch[k])
                else:
                    td_target.append(reward_batch[k] + self._discount_factor * predicted_target_q[k])

            # NB: Use td_target instead of predicted_target_q or predicted_q because it's not pure estimate (reward as samples)
            if self._normalize_returns:
                self._return_rms.update(td_target.flatten())

            # Update the critic given the targets
            predicted_q, _, ve_loss = self._value_estimator.update(
                current_state_batch, action_batch, np.reshape(td_target, (self._minibatch_size, 1)))


            # Update the actor policy using the sampled gradient
            a_outs = self._policy_estimator.predict(current_state_batch)
            grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
            self._policy_estimator.update(current_state_batch, grads[0])

            # Early stop
            if np.isnan(ve_loss):
                logger.error("Training: value estimator loss is nan, stop training")
                self._stop_training = True

            # Some basic summary of training loss
            #if self._step % 100 == 0:
            #    self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)

        # Update target networks
        self._policy_estimator.update_target_network()
        self._value_estimator.update_target_network()


class DPGAC2WithMultiPModelAndDemoAgent(DPGAC2WithDemoAgent):
    def __init__(self, sess, policy_estimator, value_estimator, model_estimator, replay_buffer, model_eval_replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise,
            summary_writer, imitation_summary_writer, model_summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1):
         super().__init__(sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise,
            summary_writer, imitation_summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates)
         self._model_summary_writer = model_summary_writer
         self._model_estimator = model_estimator
         self._model_eval_replay_buffer = model_eval_replay_buffer
         #TODO:
         self._best_action_from_model = False
         # The episode number from which use policy solely to obtain best action
         self._stop_best_action_from_model_episode = 100
         # TODO: Number of episodes sampled from real environment

         # Number of random actions to take
         self._num_random_action = 100
         # Planning horizon for model-based learning
         self._model_plan_horizon = 5
         # (global) model training number, used for writing summaries
         # TODO: Make number of model training steps a tf variable in order to preserve between sessions
         self._model_train_step = 0


    def _trainModelFull(self, epochs):
        """
        TODO: Don't iterate over whole rb as the training time increases over time as rb expands
        """
        for i in range(epochs):
            for exp in self._sampleBatch(self._minibatch_size):
                current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch = exp
                model_loss = self._model_estimator.update(current_state_batch, action_batch, next_state_batch - current_state_batch)
                self._model_summary_writer.writeSummary({
                    "ModelTrainLoss": model_loss
                    }, self._model_train_step)
            self._model_train_step += 1

    def _trainModel(self, steps):
        for i in range(steps):
            exp = self._replay_buffer.sample_batch(self._minibatch_size)
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch = exp
            _, _, model_loss = self._model_estimator.update(current_state_batch, action_batch, next_state_batch - current_state_batch)
            if i % 100 == 0:
                eval_exp = self._model_eval_replay_buffer.sample_batch(self._model_eval_replay_buffer.size())
                current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch = eval_exp
                model_eval_loss = self._model_estimator.evaluate(current_state_batch, action_batch, next_state_batch - current_state_batch)
                self._model_summary_writer.writeSummary({
                    "ModelTrainLoss": model_loss,
                    "ModelEvaluationLoss": model_eval_loss[0]
                    }, self._model_train_step)
            self._model_train_step += 1


    def _trainPolicyCriticFull(self, epochs):
        """
        TODO: Don't iterate over whole rb as the training time increases over time as rb expands
        """
        for i in range(epochs):
            for exp in self._sampleBatch(self._minibatch_size):
                current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch = exp
                current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
                action_batch = action_batch.reshape(self._minibatch_size, -1)
                reward_batch = reward_batch.reshape(self._minibatch_size, -1)
                next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
                # Calculate targets
                predicted_target_q = self._value_estimator.predict_target(
                    next_state_batch, self._policy_estimator.predict_target(next_state_batch))

                td_target = []
                for k in range(self._minibatch_size):
                    if termination_batch[k]:
                        td_target.append(reward_batch[k])
                    else:
                        td_target.append(reward_batch[k] + self._discount_factor * predicted_target_q[k])

                # Update the critic given the targets
                predicted_q, _, ve_loss = self._value_estimator.update(
                    current_state_batch, action_batch, np.reshape(td_target, (self._minibatch_size, 1)))


                # Update the actor policy using the sampled gradient
                a_outs = self._policy_estimator.predict(current_state_batch)
                grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
                self._policy_estimator.update(current_state_batch, grads[0])

                # Early stop
                if np.isnan(ve_loss):
                    logger.error("Training: value estimator loss is nan, stop training")
                    self._stop_training = True

                # Some basic summary of training loss
                #if self._step % 100 == 0:
                #    self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)

            if i % 10 == 0:
                # Update target networks
                self._policy_estimator.update_target_network()
                self._value_estimator.update_target_network()

    def _getBestAction(self):
        if not self._best_action_from_model:
            return self._policy_estimator.predict(self._last_state) + self._actor_noise()
        """
        Assume reward function is given
        """
        # TODO: Remove explicit reward from explicit environment
        from Environments.VREPEnvironments import VREPPushTaskEnvironment
        best_action = None
        max_horizon_reward = -float("inf")
        current_state = np.copy(self._last_state)
        for i in range(self._num_random_action):
            horizon_reward = 0
            first_action = None
            for j in range(self._model_plan_horizon):
                # TODO: Hard coded max velocity
                action = generateRandomAction(1.0).reshape(1, -1)
                #action = self._policy_estimator.predict(self._last_state) + self._actor_noise()
                if j == 0:
                    first_action = action
                next_state = current_state + self._model_estimator.predict(current_state, action)
                horizon_reward += VREPPushTaskEnvironment.getRewards(current_state, action)
                current_state = next_state
            if best_action is None or horizon_reward > max_horizon_reward:
                best_action = first_action
                max_horizon_reward = horizon_reward
        return best_action

    def act(self, observation, last_reward, termination, episode_start_num, episode_num, episode_num_var, is_learning=False):
        self._episode_return += last_reward
        self._step += 1
        # Putting a limit on how long the a single trial can run for simplicity
        if self._step > self._max_episode_length:
            termination = True

        current_state = observation.reshape(1, -1)

        # Initial step
        if self._last_state is None:
            # Train model
            self._trainModel(1000)
            # Initialize the last state and action
            self._last_state = current_state
            best_action = self._getBestAction()
            self._last_action = best_action
            return best_action, termination

        # Store the last step
        self._replay_buffer.add(self._last_state.squeeze().copy(), self._last_action.squeeze().copy(), last_reward,
                current_state.squeeze().copy(), termination)
        if self._normalize_states:
            self._state_rms.update(np.array([self._last_state]))


        self._last_state = current_state.copy()

        if is_learning and self._replay_buffer.size() >= self._minibatch_size and not self._stop_training:
            self._train()

        best_action = self._getBestAction()

        self._last_action = best_action

        if termination:
            # Switch between getting action from model and policy, but get from policy exclusively after episode
            # _stop_best_action_from_model_episode
            # TODO:
            #if episode_num >= self._stop_best_action_from_model_episode:
            #    self._best_action_from_model = False
            #else:
            #    self._best_action_from_model = not self._best_action_from_model
            self._best_action_from_model = False

            # Record cumulative reward of trial
            self._episode_returns.append(self._episode_return)
            average = np.mean(self._episode_returns[-self._num_rewards_to_average:])

            # Update episode number variable
            self._sess.run(tf.assign(episode_num_var, episode_num))

            # Check for improvements
            if len(self._episode_returns) >= self._num_rewards_to_average and\
                (self._best_average_episode_return is None or self._best_average_episode_return < average):
                self._best_average_episode_return = average
                improve_str = '*'
            else:
                improve_str = ''

            # Log the episode summary
            log_string = "Episode {0:>5}/{1:>5} ({2:>5}/{3:>5} in this run), " +\
                         "R {4:>9.3f}, Ave R {5:>9.3f} {7}, Ave Max Q {6:>9.3f}"

            logger.info(log_string.format(episode_num, self._num_episodes + episode_start_num - 1,
                episode_num - episode_start_num + 1,
                self._num_episodes,
                self._episode_return[0], average, self._max_q / float(self._step), improve_str))

            # Checkpoint
            if is_learning:
                if improve_str == '*':
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_save_dir, is_best=True, step=episode_num, write_meta_graph=False)
                if (episode_num % self._recent_save_freq == 0 or episode_num >= self._num_episodes +\
                    episode_start_num - 1):
                    logger.info("Saving agent checkpoints")
                    self.save(self._estimator_save_dir, step=episode_num, write_meta_graph=False)
                # Save summary
                self._summary_writer.writeSummary({
                    "TotalReward": self._episode_return[0],
                    "AverageMaxQ": self._max_q / float(self._step),
                    "BestAverage": self._best_average_episode_return,
                    }, episode_num)
            # Save replay buffer
            if (not self._replay_buffer_save_dir is None) and \
                (episode_num % self._replay_buffer_save_freq == 0 or episode_num >= self._num_episodes +\
                episode_start_num - 1):
                logger.info("Saving replay buffer")
                self.saveReplayBuffer(self._replay_buffer_save_dir)

            # Check for convergence
            if self._last_average and average <= self._last_average:
                self._num_non_imp_eps += 1
            else:
                self._num_non_imp_eps = 0
            if self._num_non_imp_eps >= self._max_num_non_imp_eps:
                logger.info("Agent not improving; stop training")
                self._stop_training = True
            self._last_average = average

            # Reset for new episode
            self._episode_return = 0.0
            self._max_q = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None

        return best_action, termination

    def _train(self):
        for _ in range(self._num_updates):
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch =\
                    self._sampleBatch(self._minibatch_size)
            current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
            # Calculate targets
            predicted_target_q = self._value_estimator.predict_target(
                next_state_batch, self._policy_estimator.predict_target(next_state_batch))

            td_target = []
            for k in range(self._minibatch_size):
                if termination_batch[k]:
                    td_target.append(reward_batch[k])
                else:
                    td_target.append(reward_batch[k] + self._discount_factor * predicted_target_q[k])

            # Update the critic given the targets
            predicted_q, _, ve_loss = self._value_estimator.update(
                current_state_batch, action_batch, np.reshape(td_target, (self._minibatch_size, 1)))
            # NB: Use td_target instead of predicted_target_q or predicted_q because it's not pure estimate (reward as samples)
            if self._normalize_returns:
                self._return_rms.update(td_target.flatten())


            # Update the actor policy using the sampled gradient
            a_outs = self._policy_estimator.predict(current_state_batch)
            grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
            self._policy_estimator.update(current_state_batch, grads[0])

            # Early stop
            if np.isnan(ve_loss):
                logger.error("Training: value estimator loss is nan, stop training")
                self._stop_training = True

            # Some basic summary of training loss
            #if self._step % 100 == 0:
            #    self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)

        # Update target networks
        self._policy_estimator.update_target_network()
        self._value_estimator.update_target_network()



class DPGAC2WithPrioritizedRB(AgentBase):
    def __init__(self, sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1):
         super().__init__(sess, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates,
            log_stats_freq)
         # Beta used in prioritized rb for importance sampling
         # TODO: Remove hardcoded value
         self._replay_buffer_beta = 1.0

    def _sampleBatch(self, batch_size, **kwargs):
        beta = kwargs["beta"] if "beta" in kwargs else self._replay_buffer_beta

        return self._replay_buffer.sample_batch(batch_size=batch_size, beta=beta)

    def _train(self):
        for _ in range(self._num_updates):
            # Calculate 1-step targets
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch, weights, indexes =\
                    self._sampleBatch(self._minibatch_size, beta=self._replay_buffer_beta)
            current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            weights = weights.reshape(self._minibatch_size, -1)
            next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)

            predicted_target_q = self._value_estimator.predict_target(
                next_state_batch, self._policy_estimator.predict_target(next_state_batch))

            td_target = []
            for k in range(self._minibatch_size):
                if termination_batch[k]:
                    td_target.append(reward_batch[k])
                else:
                    td_target.append(reward_batch[k] + self._discount_factor * predicted_target_q[k])
            td_target = np.reshape(td_target, (-1, 1))

            # Calculate n-step targets
            #TODO: Remove hardcoded N
            # Calculate 10-step targets
            num_step = 10
            ns_current_state_batch, ns_action_batch, ns_reward_batch, ns_next_state_batch, ns_termination_batch,\
                    ns_weights, ns_indexes =\
                        self._replay_buffer.sample_episode(self._replay_buffer_beta)
            ns_current_state_batch = ns_current_state_batch.reshape(-1, self._value_estimator._state_dim)
            ns_action_batch = ns_action_batch.reshape(-1, self._value_estimator._action_dim)
            ns_reward_batch = ns_reward_batch.reshape(-1, 1)
            ns_next_state_batch = ns_next_state_batch.reshape(-1, self._value_estimator._state_dim)
            ns_weights = ns_weights.reshape(-1, 1)
            rollout_length = ns_next_state_batch.shape[0]
            assert(rollout_length > 0)

            #TODO: Use action from the last transition instead? Since it's already available. Investigate
            # Only calculate n1mix target when the sampled rollout contains greater than num_step number of transitions
            if rollout_length > num_step:
                # Initialise ns_td_target
                # Note that start bootstrapping from the second-to-last transition, ie use estimate on the last 'current
                # state' and use pure reward for the last transition
                nb_ns_td_target = rollout_length - num_step + 1
                ns_td_target = np.zeros((nb_ns_td_target, 1))
                ns_td_target[:-1] = self._value_estimator.predict_target(
                        ns_next_state_batch[num_step-1:rollout_length-1],
                        self._policy_estimator.predict_target(ns_next_state_batch[num_step-1:rollout_length-1]))
                #print("NS_TD_TARGET")
                #print(ns_td_target)
                for k in range(num_step):
                    ns_td_target = ns_reward_batch[num_step-1-k:rollout_length-k] + self._discount_factor * ns_td_target
                # Combine 1-step batches with n-step batches
                current_state_batch = np.concatenate([current_state_batch, ns_current_state_batch[:nb_ns_td_target, :]], axis=0)
                action_batch = np.concatenate([action_batch, ns_action_batch[:nb_ns_td_target, :]], axis=0)
                td_target = np.concatenate([td_target, ns_td_target], axis=0)
                weights = np.concatenate([weights, ns_weights[:nb_ns_td_target, :]], axis=0)
                indexes = indexes + ns_indexes[:nb_ns_td_target]
            else:
                nb_ns_td_target = 0

            #print("WEIGHTS")
            #print(weights)
            #print("INDEXES")
            #print(indexes)
            #print("TD_TARGET")
            #print(td_target)
            #print("ROLLOUT_LENGTH")
            #print(rollout_length)
            #print(current_state_batch.shape[0])
            #print(action_batch.shape[0])
            #print(td_target.shape[0])
            #print(weights.shape[0])
            #print(len(indexes))
            # Update the critic given the targets with weights
            _, td_error, ve_weighted_loss, ve_loss = self._value_estimator.update_with_weights_and_n1s_td(
                current_state_batch, action_batch, td_target, weights, rollout_length)
            self._stats_epoch_critic_loss.append(ve_weighted_loss)

            # NB: Use td_target because it's not pure estimate (reward as samples)
            if self._normalize_returns:
                self._return_rms.update(td_target.flatten())


            # Update the actor policy using the sampled gradient
            a_outs = self._policy_estimator.predict(current_state_batch)
            grads = self._value_estimator.action_gradients(current_state_batch, a_outs)
            self._policy_estimator.update(current_state_batch, grads[0])

            # Calculate and update new priorities for sampled transitions
            #TODO: Remove hardcoded value
            lambda3 = 0.1
            epislon = 1e-3
            priorities = np.square(td_error) + lambda3 * np.square(np.linalg.norm(grads)) + epislon
            #print("TDERROR!!!")
            #print(self._replay_buffer._it_sum.sum())
            #print(self._replay_buffer._it_min.min())

            #print("ahdaowdaiwododhawido")
            #print(priorities.shape)
            #print(len(indexes))
            self._replay_buffer.update_priorities(indexes, priorities.flatten())


            # Early stop
            if np.isnan(ve_weighted_loss):
                logger.error("Training: value estimator loss is nan, stop training")
                self._stop_training = True

        # Update target networks
        self._policy_estimator.update_target_network()
        self._value_estimator.update_target_network()

