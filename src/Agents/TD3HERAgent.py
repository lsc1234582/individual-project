import numpy as np
import random
import tensorflow as tf
import time
import pprint
from Utils import ReplayBuffer
from Utils import getModuleLogger
from Utils import generateRandomAction
from Agents.DPGAC2Agent import AgentBase
import pickle as pk

# TODO: Remove after debug
#np.set_printoptions(threshold=np.nan, linewidth=200)

# Module logger
logger = getModuleLogger(__name__)

class TD3HERAgent(AgentBase):
    def __init__(self, sess, env, policy_estimator, value_estimator1, value_estimator2, replay_buffer,
            discount_factor, num_train_steps, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1, train_freq=1, eval_replay_buffer=None, test_freq=50, num_test_eps=20,
            policy_and_target_update_freq = 2):

        super().__init__(sess, env, replay_buffer,
            discount_factor, num_train_steps, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, normalize_returns, return_rms, num_updates,
            log_stats_freq, train_freq, eval_replay_buffer, test_freq, num_test_eps)

        self._policy_estimator = policy_estimator
        self._value_estimator1 = value_estimator1
        self._value_estimator2 = value_estimator2
        self._stats_epoch_Q = []
        self._stats_epoch_critic_loss = []
        #TODO: Remove DEBUG
        self._debug_freq = 200
        # Beta used in prioritized rb for importance sampling
        # TODO: Remove hardcoded value
        self._replay_buffer_beta = 1.0
        #TODO: Remove DEBUG
        self._debug_freq = 200
        self._policy_and_target_update_freq = policy_and_target_update_freq

        # For HER goal relabling
        self._episode_experience = []

    def _initialize(self):
        self._policy_estimator.update_target_network(tau=1.0)
        self._value_estimator1.update_target_network(tau=1.0)
        self._value_estimator2.update_target_network(tau=1.0)

    def _sampleBatch(self, rb, batch_size, **kwargs):
        beta = kwargs["beta"] if "beta" in kwargs else self._replay_buffer_beta

        return rb.sample_batch(batch_size=batch_size, beta=beta)


    def _getStats(self):
        # Auxiliary data to pass to child class
        aux = {}
        # Agent stats
        if self._stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self._stats_sample = self._sampleBatch(self._replay_buffer, self._stats_sample_size)
        # Combine states and goals
        #batch_size = self._stats_sample[0].shape[0]
        #print("HOHO")
        #print(len(self._stats_sample))
        #print(self._stats_sample[0].shape)
        #print(self._stats_sample[5])
        state_goal_sample = np.concatenate([self._stats_sample[0], self._stats_sample[5]], axis=1)
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
        stats_sample_action = self._getBestAction(state_goal_sample)
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

        #stats, aux = super()._getStats()

        # Agent stats
        stats_sample_q1 = self._value_estimator1.predict(state_goal_sample, self._stats_sample[1])
        stats_sample_q2 = self._value_estimator2.predict(state_goal_sample, self._stats_sample[1])
        stats["agent_sample_Q_mean"] = np.mean(stats_sample_q1)
        stats["agent_sample_Q_std"] = np.std(stats_sample_q1)
        stats["agent_sample_Q2_mean"] = np.mean(stats_sample_q2)
        stats["agent_sample_Q2_std"] = np.std(stats_sample_q2)
        stats_sample_action_q = self._value_estimator1.predict(state_goal_sample, aux["stats_sample_action"])
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

        batch_size = state.shape[0]
        goal_batch = np.array([self._env._goal.copy() for _ in range(batch_size)])

        state_goal = np.concatenate([state, goal_batch], axis=1)
        best_action_q = self._value_estimator1.predict(state_goal, best_action)
        self._stats_epoch_actions.append(best_action)
        self._stats_epoch_Q.append(best_action_q)

    def _train(self):
        for _ in range(self._num_updates):
            # Calculate 1-step targets
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch, goal_batch, weights, indexes =\
                    self._sampleBatch(self._replay_buffer, self._minibatch_size, beta=self._replay_buffer_beta)
            #current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            #action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            #next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
            #goal_batch = goal_batch.reshape(self._minibatch_size, -1)
            weights = weights.reshape(self._minibatch_size, -1)
            #print("HIYO")
            #print(current_state_batch.shape)
            #print(goal_batch.shape)
            current_state_goal_batch = np.concatenate([current_state_batch, goal_batch], axis=1)

            next_state_goal_batch = np.concatenate([next_state_batch, goal_batch], axis=1)

            target_action = self._policy_estimator.predict_target(next_state_goal_batch)
            target_action += np.clip(np.random.normal(0.0, 0.02, size=target_action.shape), -0.05, 0.05)
            predicted_target_q1 = self._value_estimator1.predict_target(
                next_state_goal_batch, target_action)
            predicted_target_q2 = self._value_estimator2.predict_target(
                next_state_goal_batch, target_action)
            min_predicted_target_q = np.min([predicted_target_q1, predicted_target_q2], axis=0)

            td_target = np.copy(reward_batch)
            td_target[~termination_batch, :] += self._discount_factor * min_predicted_target_q[~termination_batch, :]

            # Update the critic given the targets
            _, td_error, ve_weighted_loss, ve_loss = self._value_estimator1.update_with_weights(
                    current_state_goal_batch, action_batch, td_target, weights)
            _, td_error2, ve_weighted_loss2, ve_loss2 = self._value_estimator2.update_with_weights(
                    current_state_goal_batch, action_batch, td_target, weights)
            self._stats_epoch_critic_loss.append(ve_loss)

            # NB: Use td_target because it's not pure estimate (reward as samples)
            if self._normalize_returns:
                self._return_rms.update(td_target.flatten())

            a_outs = self._policy_estimator.predict(current_state_goal_batch)
            grads = self._value_estimator1.action_gradients(current_state_goal_batch, a_outs)

            # Calculate and update new priorities for sampled transitions
            #TODO: Remove hardcoded value
            lambda3 = 0.1
            epislon = 1e-2
            demo_bonuses = np.zeros_like(td_error)
            demo_bonuses[np.array(indexes) < self._replay_buffer.get_loaded_storage_size(), :] = 0.05
            #print("HAHA")
            #print(demo_bonuses)
            #print(td_error.shape[0] - nb_ns_td_target)
            priorities = np.square(td_error) + lambda3 * np.square(np.linalg.norm(grads)) + epislon + demo_bonuses
            #print("TDERROR!!!")
            #print(self._replay_buffer._it_sum.sum())
            #print(self._replay_buffer._it_min.min())
            assert(not np.isnan(self._replay_buffer._it_sum.sum()))
            assert(not np.isinf(self._replay_buffer._it_sum.sum()))
            assert(self._replay_buffer._it_sum.sum() > 0)
            assert(not np.isnan(self._replay_buffer._it_min.min()))
            assert(not np.isinf(self._replay_buffer._it_min.min()))
            assert(self._replay_buffer._it_min.min() > 0)

            #TODO: Remove DEBUG
            #if self._stats_tot_steps % self._debug_freq == 0:
            #    with open("DEBUG_rb_freq.pkl", "wb") as f:
            #        pk.dump(self._replay_buffer.debug_get_sample_freq(), f)
            #    with open("DEBUG_rb_priorities.pkl", "wb") as f:
            #        pk.dump(self._replay_buffer.debug_get_priorities(), f)

            # Early stop
            if np.isnan(ve_loss) or np.isnan(ve_loss2):
                logger.error("Training: value estimator loss is nan, stop training")
                self._stop_training = True

            # Some basic summary of training loss
            #if self._step % 100 == 0:
            #    self._summary_writer.writeSummary({"ValueEstimatorTrainLoss": ve_loss}, self._step)
                # Update target networks and policy
            if self._stats_tot_steps % self._policy_and_target_update_freq == 0:
                # Update the actor policy using the sampled gradient
                self._policy_estimator.update(current_state_goal_batch, grads[0])
                self._policy_estimator.update_target_network()
                self._value_estimator1.update_target_network()
                self._value_estimator2.update_target_network()

    def _evaluate(self):
        pass

    def _normalizationUpdateAfterRB(self, *args, **kwargs):
        if self._normalize_states:
            self._state_rms.update(np.array(np.concatenate([self._last_state, self._env._goal.reshape(1, -1)],
                axis=1)))


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
            best_action = self._getBestAction(np.concatenate([self._last_state, self._env._goal.reshape(1, -1)], axis=1))
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
                current_state.squeeze().copy(), termination, self._env._goal.copy())
            self._episode_experience.append((self._last_state.squeeze().copy(), self._last_action.squeeze().copy(),
                last_reward.squeeze(),
                current_state.squeeze().copy(), termination, self._env._goal.copy()))
            # TODO: Should normalize for test episodes as well?
            self._normalizationUpdateAfterRB(current_state=current_state)

        if not termination:
            self._last_state = current_state.copy()
            best_action = self._getBestAction(np.concatenate([self._last_state, self._env._goal.reshape(1, -1)], axis=1))
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
                # Add goal re-labelled experiences to replay buffer
                episode_num_this_run = episode_num - episode_start_num + 1
                # Make the last state reached in the episode the new goal
                _, _, _, last_State, _, _ = self._episode_experience[-1]
                new_goal = self._env.extractGoal(last_State)
                for i in range(len(self._episode_experience)):
                    s, a, _, ns, d, _ = self._episode_experience[i]
                    r = self._env.getRewards(s, a, ns, goal=new_goal)
                    #print("REPLAY BUFFER")
                    #print(self._env.getStateString(s))
                    #print(a)
                    #print(r)
                    #print(self._env.getStateString(ns))
                    #print(new_goal)
                    self._replay_buffer.add(s, a, r.squeeze(), ns, d, new_goal.squeeze().copy())

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
                # Check for convergence
                #if self._last_average and average <= self._last_average:
                #    self._num_non_imp_eps += 1
                #else:
                #    self._num_non_imp_eps = 0
                #if self._num_non_imp_eps >= self._max_num_non_imp_eps:
                #    logger.info("Agent is not improving; stop training")
                #    self._stop_training = True
                #self._last_average = average

            # Reset for new episode
            self._episode_return = 0.0
            self._episode_return_dense = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None
            self.reset()
            self._episode_experience = []

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
