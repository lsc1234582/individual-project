import numpy as np
import random
import tensorflow as tf
import time
import pprint
from Utils import ReplayBuffer
from Utils import getModuleLogger
from Utils import generateRandomAction
import pickle as pk

# TODO: Remove after debug
np.set_printoptions(threshold=np.nan, linewidth=200)

# Module logger
logger = getModuleLogger(__name__)

class AgentBase(object):
    def __init__(self, sess, env, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1, train_freq=1, eval_replay_buffer=None, test_freq=50, num_test_eps=20):
        self._sess = sess
        self._env = env
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
        self._stats_sample_size = 100

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

        self._eval_replay_buffer = eval_replay_buffer

        self._normalize_states = normalize_states
        self._state_rms = state_rms
        self._normalize_returns = normalize_returns
        self._return_rms = return_rms

        # Number of updates per training step
        self._num_updates = num_updates
        # Training frequency in number of rollout steps
        self._train_freq = train_freq

        # Test for success rate
        # Test frequency, in number of episodes
        self._is_test_episode = False
        self._test_freq = test_freq
        self._num_test_eps = num_test_eps
        self._num_success_test_eps = 0
        self._last_success_rate = 0.0
        self._test_eps_counter = 0


    def _sampleBatch(self, rb, batch_size, **kwargs):
        return rb.sample_batch(batch_size=batch_size)

    def _getStats(self):
        # Auxiliary data to pass to child class
        aux = {}
        # Agent stats
        if self._stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self._stats_sample = self._sampleBatch(self._replay_buffer, self._stats_sample_size)
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
        stats_sample_action = self._getBestAction(self._stats_sample[0])
        aux["stats_sample_action"] = stats_sample_action
        stats["agent_sample_action_mean"] = np.mean(stats_sample_action)
        stats["agent_sample_action_std"] = np.std(stats_sample_action)

        # Epoch stats
        stats_tot_duration = time.time() - self._stats_start_time
        stats['epoch/episode_return'] = np.mean(self._stats_epoch_episode_returns)
        stats['epoch/episode_steps'] = np.mean(self._stats_epoch_episode_steps)
        # TODO: Again, full action vector instead of reduced value?
        stats['epoch/actions_mean'] = np.mean(self._stats_epoch_actions)
        stats['epoch/actions_std'] = np.std(self._stats_epoch_actions)
        if self._test_freq > 0:
            stats["test/success_rate"] = self._last_success_rate
        # Clear epoch statistics.
        self._stats_epoch_episode_returns = []
        self._stats_epoch_episode_steps = []
        self._stats_epoch_actions = []

        # Evaluation statistics.
        #if eval_env is not None:
        #    stats['epoch/eval/episode_return'] = np.mean(self._epoch_eval_episode_return)
        #    stats['epoch/eval/Q_mean'] = np.mean(self._epoch_eval_Q_mean)
        #    stats['epoch/eval/Q_std'] = np.std(self._epoch_eval_Q_mean)

        # Total statistics.
        stats["total/score"] = self.score()
        stats['total/duration'] = stats_tot_duration
        stats['total/steps_per_second'] = float(self._stats_tot_steps) / float(stats_tot_duration)
        stats['total/steps'] = self._stats_tot_steps

        return stats, aux

    def _logStats(self, episode_num):
        """
        Logging happens at the end of every logging epoch, which is determined by the logging frequency, in number of
        episodes
        """
        stats, _ = self._getStats()
        #pprint.pprint(stats)
        self._summary_writer.writeSummary(stats, episode_num)


    def initialize(self):
        """
        Target networks share the same parameters with the behaviourial networks at the beginning
        """
        logger.info("Initializing agent {}".format(self.__class__.__name__))
        self._sess.run(tf.global_variables_initializer())
        self._initialize()


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
        assert self._replay_buffer is not None
        self._replay_buffer.load(path)
        logger.info("Replay buffer loaded")

    def loadEvalReplayBuffer(self, path):
        assert self._eval_replay_buffer is not None
        self._eval_replay_buffer.load(path)
        logger.info("Evaluation replay buffer loaded")

    def score(self):
        return self._best_average_episode_return

    def reset(self):
        self._actor_noise.reset()

    def _normalizationUpdateAfterRB(self, *args, **kwargs):
        if self._normalize_states:
            self._state_rms.update(np.array([self._last_state]))


    def act(self, current_state, last_reward, termination, episode_start_num, episode_num, episode_num_var, is_learning=False):
        """
        NB: Although the shape of the inputs are all batch version, this function only deals with single-step
        transition. This means the "batch_size" is always 1

        Args
        -------
        observation:        array(state_dim)/array(-1, state_dim)
        last_reward:        array(1)/array(-1, 1)
        termination:        Boolean
        episode_start_num:  Int
        episode_num:        Int                                     The training episode number
        episode_num_var:    tf.Variable
        is_learning:        Boolean

        Returns
        -------
        best_action:        array(-1, action_dim)
        termination:        Boolean
        """
        current_state = current_state.reshape(1, -1)
        # Initialize the last state and action
        if self._last_state is None:
            self._last_state = current_state
            best_action = self._getBestAction(self._last_state)
            # Add exploration noise when training
            if is_learning:
                best_action += self._actor_noise()
            self._last_action = best_action
            if not self._is_test_episode:
                self._recordLogAfterBestAction(state=self._last_state, best_action=best_action)
            return best_action, termination, self._is_test_episode

        # Book keeping
        self._episode_return += last_reward.squeeze()
        self._step += 1
        # Set a limit on how long each episode is, despite what the environment responds.
        if self._step >= self._max_episode_length:
            termination = True
        if not self._is_test_episode:
            self._stats_tot_steps += 1
            # Store the last step
            self._replay_buffer.add(self._last_state.squeeze().copy(), self._last_action.squeeze().copy(),
                last_reward.squeeze(),
                current_state.squeeze().copy(), termination)
            # TODO: Should normalize for test episodes as well?
            self._normalizationUpdateAfterRB(current_state=current_state)

        if not termination:
            self._last_state = current_state.copy()
            best_action = self._getBestAction(self._last_state)
            # Add exploration noise when training
            if is_learning:
                best_action += self._actor_noise()
            if not self._is_test_episode:
                self._recordLogAfterBestAction(state=self._last_state, best_action=best_action)

            self._last_action = best_action
        else:
            if self._is_test_episode:
                self._test_eps_counter += 1
                if self._env._reachedGoalState(current_state):
                    self._num_success_test_eps += 1

                if self._test_eps_counter >= self._num_test_eps:
                    self._last_success_rate = float(self._num_success_test_eps) / float(self._num_test_eps)
                    logger.info("Success rate: {}/{}.".format(self._num_success_test_eps, self._num_test_eps))
                    self._num_success_test_eps = 0
                    self._test_eps_counter = 0
                    self._is_test_episode = False
            else:
                episode_num_this_run = episode_num - episode_start_num + 1
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
                    episode_num_this_run,
                    self._num_episodes,
                    self._episode_return, average, improve_str))


                # Checkpoint
                if is_learning:
                    if improve_str == '*':
                        logger.info("Saving best agent so far")
                        self.save(self._estimator_save_dir, is_best=True, step=episode_num, write_meta_graph=False)
                    if (episode_num_this_run % self._recent_save_freq == 0 or episode_num_this_run >= self._num_episodes):
                        logger.info("Saving agent checkpoints")
                        self.save(self._estimator_save_dir, step=episode_num, write_meta_graph=False)

                # Save replay buffer
                if (not self._replay_buffer_save_dir is None) and \
                    (episode_num_this_run % self._replay_buffer_save_freq == 0 or episode_num_this_run >= self._num_episodes):
                    logger.info("Saving replay buffer")
                    self.saveReplayBuffer(self._replay_buffer_save_dir)

                # Check for convergence
                #if self._last_average and average <= self._last_average:
                #    self._num_non_imp_eps += 1
                #else:
                #    self._num_non_imp_eps = 0
                #if self._num_non_imp_eps >= self._max_num_non_imp_eps:
                #    logger.info("Agent is not improving; stop training")
                #    self._stop_training = True
                #self._last_average = average
                # Check if should switch to testing
                if self._test_freq > 0 and episode_num_this_run % self._test_freq == 0:
                    logger.info("Start testing.")
                    self._is_test_episode = True

            # Reset for new episode
            self._episode_return = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None
            self.reset()


        # Log stats
        if (not self._is_test_episode) and self._log_stats_freq > 0 and self._stats_tot_steps % self._log_stats_freq == 0:
            self._logStats(self._stats_tot_steps)

        if (not self._is_test_episode) and self._stats_tot_steps % self._train_freq == 0:
            # Train step
            if is_learning and self._replay_buffer.size() >= self._minibatch_size and not self._stop_training:
                self._train()

            # Evaluate step
            if self._eval_replay_buffer is not None and self._eval_replay_buffer.size() >= self._minibatch_size:
                self._evaluate()

        if not termination:
            return best_action, termination, self._is_test_episode
        else:
            return None, termination, self._is_test_episode

class DPGAC2Agent(AgentBase):
    def __init__(self, sess, env, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1, train_freq=1, eval_replay_buffer=None, test_freq=50, num_test_eps=20):
        super().__init__(sess, env, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates,
            log_stats_freq, train_freq, eval_replay_buffer, test_freq, num_test_eps)

        self._policy_estimator = policy_estimator
        self._value_estimator = value_estimator
        self._stats_epoch_Q = []
        self._stats_epoch_critic_loss = []
        #TODO: Remove DEBUG
        self._debug_freq = 200

    def _initialize(self):
        self._policy_estimator.update_target_network(tau=1.0)
        self._value_estimator.update_target_network(tau=1.0)


    def _getStats(self):
        stats, aux = super()._getStats()

        # Agent stats
        stats_sample_q = self._value_estimator.predict(self._stats_sample[0], self._stats_sample[1])
        stats["agent_sample_Q_mean"] = np.mean(stats_sample_q)
        stats["agent_sample_Q_std"] = np.std(stats_sample_q)
        stats_sample_action_q = self._value_estimator.predict(self._stats_sample[0], aux["stats_sample_action"])
        stats["agent_sample_action_Q_mean"] = np.mean(stats_sample_action_q)
        stats["agent_sample_action_Q_std"] = np.std(stats_sample_action_q)

        # Epoch stats
        stats['epoch/Q_mean'] = np.mean(self._stats_epoch_Q)
        stats['epoch/Q_std'] = np.std(self._stats_epoch_Q)
        #stats['epoch/actor_loss'] = np.mean(self._stats_epoch_actor_loss)
        stats['epoch/critic_loss'] = np.mean(self._stats_epoch_critic_loss)
        # Clear epoch statistics.
        self._stats_epoch_Q = []
        self._stats_epoch_critic_loss = []

        return stats, aux

    def _getBestAction(self, state):
        best_action = self._policy_estimator.predict(state)
        return best_action

    def _recordLogAfterBestAction(self, *args, **kwargs):
        """
        Record various metric for stats purpose
        """
        state = kwargs["state"]
        best_action = kwargs["best_action"]
        best_action_q = self._value_estimator.predict(state, best_action)
        self._stats_epoch_actions.append(best_action)
        self._stats_epoch_Q.append(best_action_q)

    def _train(self):
        for _ in range(self._num_updates):
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch =\
                    self._sampleBatch(self._replay_buffer, self._minibatch_size)
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

            #TODO: Remove DEBUG
            if self._stats_tot_steps % self._debug_freq == 0:
                with open("DEBUG_rb_freq.pkl", "wb") as f:
                    pk.dump(self._replay_buffer.debug_get_sample_freq(), f)

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

    def _evaluate(self):
        pass


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
            for exp in self._sampleBatch(self._replay_buffer, self._minibatch_size):
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
            for exp in self._sampleBatch(self._replay_buffer, self._minibatch_size):
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
            #if self._last_average and average <= self._last_average:
            #    self._num_non_imp_eps += 1
            #else:
            #    self._num_non_imp_eps = 0
            #if self._num_non_imp_eps >= self._max_num_non_imp_eps:
            #    logger.info("Agent not improving; stop training")
            #    self._stop_training = True
            #self._last_average = average

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
                    self._sampleBatch(self._replay_buffer, self._minibatch_size)
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
            for exp in self._sampleBatch(self._replay_buffer, self._minibatch_size):
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
            for exp in self._sampleBatch(self._replay_buffer, self._minibatch_size):
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
            #if self._last_average and average <= self._last_average:
            #    self._num_non_imp_eps += 1
            #else:
            #    self._num_non_imp_eps = 0
            #if self._num_non_imp_eps >= self._max_num_non_imp_eps:
            #    logger.info("Agent not improving; stop training")
            #    self._stop_training = True
            #self._last_average = average

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
                    self._sampleBatch(self._replay_buffer, self._minibatch_size)
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



class DPGAC2WithPrioritizedRB(DPGAC2Agent):
    def __init__(self, sess, env, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1, train_freq=1, eval_replay_buffer=None, test_freq=50, num_test_eps=20):
         super().__init__(sess, env, policy_estimator, value_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates,
            log_stats_freq, train_freq, eval_replay_buffer, test_freq, num_test_eps)
         # Beta used in prioritized rb for importance sampling
         # TODO: Remove hardcoded value
         self._replay_buffer_beta = 1.0
         #TODO: Remove DEBUG
         self._debug_freq = 200


    def _sampleBatch(self, rb, batch_size, **kwargs):
        beta = kwargs["beta"] if "beta" in kwargs else self._replay_buffer_beta

        return rb.sample_batch(batch_size=batch_size, beta=beta)

    def _train(self):
        pp = pprint.PrettyPrinter(width=200, compact=True)

        for _ in range(self._num_updates):
            # Calculate 1-step targets
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch, weights, indexes =\
                    self._sampleBatch(self._replay_buffer, self._minibatch_size, beta=self._replay_buffer_beta)
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

            _, td_error, ve_weighted_loss, ve_loss = self._value_estimator.update_with_weights(
                    current_state_batch, action_batch, td_target, weights)

            self._stats_epoch_critic_loss.append(ve_weighted_loss)
            #print("TD_ERROR")
            #print(td_error)
            #print("VE_WEIGHTED_LOSS")
            #print(ve_weighted_loss)
            #print("VE_LOSS")
            #print(ve_loss)

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
            epislon = 1e-2
            demo_bonuses = np.zeros_like(td_error)
            demo_bonuses[np.array(indexes) < self._replay_buffer.get_loaded_storage_size(), :] = 1.0
            #print("HAHA")
            #print(demo_bonuses)
            #print(td_error.shape[0] - nb_ns_td_target)
            priorities = np.square(td_error) + lambda3 * np.square(np.linalg.norm(grads)) + epislon + demo_bonuses
            #print("TDERROR!!!")
            #print(self._replay_buffer._it_sum.sum())
            #print(self._replay_buffer._it_min.min())
            assert(not np.isnan(self._replay_buffer._it_sum.sum()))
            assert(not np.isnan(self._replay_buffer._it_min.min()))

            #TODO: Remove DEBUG
            if self._stats_tot_steps % self._debug_freq == 0:
                with open("DEBUG_rb_freq.pkl", "wb") as f:
                    pk.dump(self._replay_buffer.debug_get_sample_freq(), f)
                with open("DEBUG_rb_priorities.pkl", "wb") as f:
                    pk.dump(self._replay_buffer.debug_get_priorities(), f)

            #pp = pprint.PrettyPrinter(width=200, compact=True)
            #print("Priorities")
            ##print(priorities.shape)
            #print("TMP_STATS:----------")
            #print(len(indexes))
            #tmp_stats = np.concatenate([
            #    np.reshape(indexes, (-1, 1)),
            #    np.reshape(td_target, (-1, 1)),
            #    np.reshape(td_error, (-1, 1)),
            #    priorities,
            #    np.reshape(weights, (-1, 1)),
            #    ], axis=1)
            #pp.pprint(tmp_stats)
            #pp.pprint(ve_weighted_loss)
            #pp.pprint(ve_loss)
            self._replay_buffer.update_priorities(indexes, priorities.flatten())


            # Early stop
            if np.isnan(ve_weighted_loss):
                logger.error("Training: value estimator loss is nan, stop training")
                self._stop_training = True

            # Update target networks
            self._policy_estimator.update_target_network()
            self._value_estimator.update_target_network()

class ModelBasedAgent(AgentBase):
    def __init__(self, sess, env, model_estimator, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, state_change_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1, train_freq=1, eval_replay_buffer=None, test_freq=50, num_test_eps=20):
        super().__init__(sess, env, replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates,
            log_stats_freq, train_freq, eval_replay_buffer, test_freq, num_test_eps)

        self._model_estimator = model_estimator

        self._stats_epoch_Q = []
        self._stats_epoch_model_loss = []
        if self._eval_replay_buffer is not None:
            self._stats_epoch_model_eval_loss = []
            #TODO: remove hard coded
            self._num_evals = 10

        #TODO: remove hard coded
        # Number of random actions to take
        self._num_random_action = 100
        # Planning horizon for model-based learning
        self._model_plan_horizon = 5
        self._state_change_rms = state_change_rms


    def _initialize(self):
        pass

    def _getStats(self):
        stats, aux = super()._getStats()
        # Agent stats
        if self._normalize_states:
            ops = [tf.reduce_mean(self._state_change_rms.mean), tf.reduce_mean(self._state_change_rms.std)]
            values = self._sess.run(ops)
            stats["state_change_rms_mean"] = values[0]
            stats["state_change_rms_std"] = values[1]
        # Epoch stats
        stats['epoch/model_loss'] = np.mean(self._stats_epoch_model_loss)
        if self._eval_replay_buffer is not None:
            stats['epoch/model_eval_loss'] = np.mean(self._stats_epoch_model_eval_loss)
        # Clear epoch statistics.
        self._stats_epoch_model_loss = []
        if self._eval_replay_buffer is not None:
            self._stats_epoch_model_eval_loss = []

        return stats, aux

    def _getBestAction(self, state):
        """
        Assume reward function is given

        Args
        -------
        state:          array(state_dim)/array(-1, state_dim)

        Returns
        best_action:    array(-1, action_dim)
        -------
        """
        assert self._num_random_action != 0
        assert self._model_plan_horizon != 0
        state = state.reshape(-1, self._model_estimator._state_dim)
        batch_size = state.shape[0]

        best_action = None
        max_horizon_reward = np.zeros((batch_size, 1)) - float("inf")
        for i in range(self._num_random_action):
            first_action = None
            horizon_reward = np.zeros((batch_size, 1))
            current_state = np.copy(state)
            for j in range(self._model_plan_horizon):
                # TODO: Remove hard coded max velocity
                action = generateRandomAction(1.0, 7 * batch_size).reshape(batch_size, -1)
                if j == 0:
                    first_action = action
                # Proceed to next state
                next_state = current_state + self._model_estimator.predict(current_state, action)
                horizon_reward += self._env.getRewards(current_state, action, next_state)
                current_state = np.copy(next_state)
            if best_action is None:
                best_action = first_action
                max_horizon_reward = horizon_reward
            else:
                better_action_indx = (horizon_reward > max_horizon_reward).squeeze()
                best_action[better_action_indx, :] = first_action[better_action_indx, :]
                max_horizon_reward[better_action_indx, :] = horizon_reward[better_action_indx, :]

        return best_action

    def _recordLogAfterBestAction(self, *args, **kwargs):
        """
        Record various metric for stats purpose
        """
        best_action = kwargs["best_action"]
        self._stats_epoch_actions.append(best_action)

    def _normalizationUpdateAfterRB(self, *args, **kwargs):
        if self._normalize_states:
            self._state_rms.update(np.array([self._last_state]))
            state_change = kwargs["current_state"] - self._last_state
            self._state_change_rms.update(np.array([state_change]))

    def _train(self):
        # Train
        #print("training model!!")
        #print(self._num_updates)
        for _ in range(self._num_updates):
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch =\
                    self._sampleBatch(self._replay_buffer, self._minibatch_size)
            current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)

            _, _, model_loss = self._model_estimator.update(current_state_batch, action_batch, next_state_batch - current_state_batch)
            self._stats_epoch_model_loss.append(model_loss)
            # Early stop
            if np.isnan(model_loss):
                logger.error("Training: model estimator loss is nan, stop training")
                self._stop_training = True

    def _evaluate(self):
        # Evaluation
        for _ in range(self._num_evals):
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch =\
                    self._sampleBatch(self._eval_replay_buffer, self._minibatch_size)
            current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)

            model_eval_loss = self._model_estimator.evaluate(current_state_batch, action_batch,
                    next_state_batch - current_state_batch)
            self._stats_epoch_model_eval_loss.append(model_eval_loss)

