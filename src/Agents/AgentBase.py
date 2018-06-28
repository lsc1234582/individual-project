import collections
import numpy as np
import tensorflow as tf
import time
from Utils.Utils import getModuleLogger

# Module logger
logger = getModuleLogger(__name__)

class AgentBase(object):
    def __init__(self, sess, env, replay_buffer,
            discount_factor, num_train_steps, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1, train_freq=1, eval_replay_buffer=None, test_freq=50, num_test_eps=20):
        self._sess = sess
        self._env = env
        self._discount_factor = discount_factor
        self._num_train_steps = num_train_steps
        self._max_episode_length = max_episode_length
        self._minibatch_size = minibatch_size
        self._actor_noise = actor_noise

        self._step = 0

        self._last_state = None
        self._last_action = None

        # To implement early stop
        self._stop_training = False

        # Summary and checkpoints
        # Agent stats related
        self._stats_sample = None
        self._stats_sample_size = 100

        # The number of episodes to average total reward over; used for score
        self._num_rewards_to_average = 100

        self._episode_return = 0
        self._episode_return_dense = 0
        self._episode_returns = collections.deque(maxlen=self._num_rewards_to_average)
        # Rollout stats related
        # Our objective metric
        self._best_average_episode_return = None
        self._stats_start_time = time.time()
        self._stats_epoch_episode_returns = []
        self._stats_epoch_episode_returns_dense = []
        self._stats_epoch_episode_steps = []
        self._stats_epoch_actions = []
        self._stats_tot_steps = 0
        self._log_stats_freq = log_stats_freq

        self._best_score_step = 0

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
        stats['epoch/episode_return_dense'] = np.mean(self._stats_epoch_episode_returns_dense)
        stats['epoch/episode_steps'] = np.mean(self._stats_epoch_episode_steps)
        # TODO: Again, full action vector instead of reduced value?
        stats['epoch/actions_mean'] = np.mean(self._stats_epoch_actions)
        stats['epoch/actions_std'] = np.std(self._stats_epoch_actions)
        if self._test_freq > 0:
            stats["test/success_rate"] = self._last_success_rate
        # Clear epoch statistics.
        self._stats_epoch_episode_returns = []
        self._stats_epoch_episode_returns_dense = []
        self._stats_epoch_episode_steps = []
        self._stats_epoch_actions = []

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

    def scoreStep(self):
        return self._best_score_step

    def reset(self):
        self._actor_noise.reset()

    def _normalizationUpdateAfterRB(self, *args, **kwargs):
        if self._normalize_states:
            self._state_rms.update(np.array([self._last_state]))


    def act(self, current_state, last_reward, last_reward_dense, termination, episode_start_num, episode_num, global_step_num, is_learning=False):
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
        global_step_num:    tf.vaiable
        is_learning:        Boolean

        Returns
        -------
        best_action:        array(-1, action_dim)
        termination:        Boolean
        """
        current_state = current_state.reshape(1, -1)
        switched_to_train = False
        # Initialize the last state and action
        if self._last_state is None:
            self._last_state = current_state
            best_action = self._getBestAction(self._last_state)
            if not self._is_test_episode:
                # Add exploration noise when training
                if is_learning:
                    best_action += self._actor_noise()
                self._recordLogAfterBestAction(state=self._last_state, best_action=best_action)
            self._last_action = best_action
            return best_action, termination, self._is_test_episode, self._stats_tot_steps

        # Book keeping
        self._episode_return += last_reward.squeeze()
        self._episode_return_dense += last_reward_dense.squeeze()
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
            if not self._is_test_episode:
                # Add exploration noise when training
                if is_learning:
                    best_action += self._actor_noise()
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
                    switched_to_train = True
            else:
                episode_num_this_run = episode_num - episode_start_num + 1
                # Record cumulative reward of trial
                self._episode_returns.append(self._episode_return)
                self._stats_epoch_episode_returns.append(self._episode_return)
                self._stats_epoch_episode_returns_dense.append(self._episode_return_dense)
                self._stats_epoch_episode_steps.append(self._step)
                average = np.mean(self._episode_returns)

                # Check for improvements
                if len(self._episode_returns) >= self._num_rewards_to_average and\
                    (self._best_average_episode_return is None or self._best_average_episode_return < average):
                    self._best_average_episode_return = average
                    self._best_score_step = self._stats_tot_steps
                    improve_str = '*'
                else:
                    improve_str = ''

                # Log the episode summary
                log_string = "Episode {0:>5} ({1:>5} in this run), R {2:>9.3f}, Ave R {3:>9.3f} {4}"

                logger.info(log_string.format(episode_num,
                    episode_num_this_run,
                    self._episode_return, average, improve_str))

                # Checkpoint best
                if is_learning and improve_str == '*':
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_save_dir, is_best=True, step=episode_num, write_meta_graph=False)

                # Save replay buffer
                if (self._replay_buffer_save_dir is not None) and \
                    (self._stats_tot_steps % self._replay_buffer_save_freq == 0):
                    logger.info("Saving replay buffer")
                    self.saveReplayBuffer(self._replay_buffer_save_dir)

            # Reset for new episode
            self._episode_return = 0.0
            self._episode_return_dense = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None
            self.reset()

        if (not switched_to_train) and (not self._is_test_episode):
            # Checkpoint recent
            if is_learning and self._stats_tot_steps % self._recent_save_freq == 0:
                logger.info("Saving agent checkpoints")
                self._sess.run(tf.assign(global_step_num, self._stats_tot_steps))
                self.save(self._estimator_save_dir, step=episode_num, write_meta_graph=False)

            # Log stats
            if self._log_stats_freq > 0 and self._stats_tot_steps % self._log_stats_freq == 0:
                logger.info("Logging stats at step {}".format(self._stats_tot_steps))
                self._logStats(self._stats_tot_steps)

            if self._stats_tot_steps % self._train_freq == 0:
                # Train step
                if is_learning and self._replay_buffer.size() >= self._minibatch_size and not self._stop_training:
                    self._train()

                # Evaluate step
                if self._eval_replay_buffer is not None and self._eval_replay_buffer.size() >= self._minibatch_size:
                    self._evaluate()

            # Check if should switch to testing
            if self._test_freq > 0 and self._stats_tot_steps % self._test_freq == 0:
                logger.info("Start testing.")
                self._is_test_episode = True

        if not termination:
            return best_action, termination, self._is_test_episode, self._stats_tot_steps
        else:
            return None, termination, self._is_test_episode, self._stats_tot_steps
