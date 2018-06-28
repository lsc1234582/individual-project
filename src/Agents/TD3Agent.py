import numpy as np
import random
import tensorflow as tf
from Utils.Utils import getModuleLogger
from Agents.AgentBase import AgentBase

# Module logger
logger = getModuleLogger(__name__)

class TD3Agent(AgentBase):
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
        # Beta used in prioritized rb for importance sampling
        # TODO: Remove hardcoded value
        self._replay_buffer_beta = 1.0
        self._policy_and_target_update_freq = policy_and_target_update_freq

    def _initialize(self):
        self._policy_estimator.update_target_network(tau=1.0)
        self._value_estimator1.update_target_network(tau=1.0)
        self._value_estimator2.update_target_network(tau=1.0)

    def _sampleBatch(self, rb, batch_size, **kwargs):
        beta = kwargs["beta"] if "beta" in kwargs else self._replay_buffer_beta

        return rb.sample_batch(batch_size=batch_size, beta=beta)


    def _getStats(self):
        stats, aux = super()._getStats()

        # Agent stats
        stats_sample_q1 = self._value_estimator1.predict(self._stats_sample[0], self._stats_sample[1])
        stats_sample_q2 = self._value_estimator2.predict(self._stats_sample[0], self._stats_sample[1])
        stats["agent_sample_Q_mean"] = np.mean(stats_sample_q1)
        stats["agent_sample_Q_std"] = np.std(stats_sample_q1)
        stats["agent_sample_Q2_mean"] = np.mean(stats_sample_q2)
        stats["agent_sample_Q2_std"] = np.std(stats_sample_q2)
        stats_sample_action_q = self._value_estimator1.predict(self._stats_sample[0], aux["stats_sample_action"])
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
        best_action_q = self._value_estimator1.predict(state, best_action)
        self._stats_epoch_actions.append(best_action)
        self._stats_epoch_Q.append(best_action_q)

    def _train(self):
        for _ in range(self._num_updates):
            # Calculate 1-step targets
            current_state_batch, action_batch, reward_batch, next_state_batch, termination_batch, weights, indexes =\
                    self._sampleBatch(self._replay_buffer, self._minibatch_size, beta=self._replay_buffer_beta)
            current_state_batch = current_state_batch.reshape(self._minibatch_size, -1)
            action_batch = action_batch.reshape(self._minibatch_size, -1)
            reward_batch = reward_batch.reshape(self._minibatch_size, -1)
            weights = weights.reshape(self._minibatch_size, -1)
            next_state_batch = next_state_batch.reshape(self._minibatch_size, -1)
            target_action = self._policy_estimator.predict_target(next_state_batch)
            target_action += np.clip(np.random.normal(0.0, 0.02, size=target_action.shape), -0.05, 0.05)
            predicted_target_q1 = self._value_estimator1.predict_target(
                next_state_batch, target_action)
            predicted_target_q2 = self._value_estimator2.predict_target(
                next_state_batch, target_action)
            min_predicted_target_q = np.min([predicted_target_q1, predicted_target_q2], axis=0)

            td_target = np.copy(reward_batch)
            td_target[~termination_batch, :] += self._discount_factor * min_predicted_target_q[~termination_batch, :]

            # Update the critic given the targets
            _, td_error, ve_weighted_loss, ve_loss = self._value_estimator1.update_with_weights(
                    current_state_batch, action_batch, td_target, weights)
            _, td_error2, ve_weighted_loss2, ve_loss2 = self._value_estimator2.update_with_weights(
                    current_state_batch, action_batch, td_target, weights)
            self._stats_epoch_critic_loss.append(ve_loss)

            # NB: Use td_target because it's not pure estimate (reward as samples)
            if self._normalize_returns:
                self._return_rms.update(td_target.flatten())

            a_outs = self._policy_estimator.predict(current_state_batch)
            grads = self._value_estimator1.action_gradients(current_state_batch, a_outs)

            # Calculate and update new priorities for sampled transitions
            #TODO: Remove hardcoded value
            lambda3 = 0.1
            epislon = 1e-2
            demo_bonuses = np.zeros_like(td_error)
            demo_bonuses[np.array(indexes) < self._replay_buffer.get_loaded_storage_size(), :] = 0.05
            priorities = np.square(td_error) + lambda3 * np.square(np.linalg.norm(grads)) + epislon + demo_bonuses
            assert(not np.isnan(self._replay_buffer._it_sum.sum()))
            assert(not np.isinf(self._replay_buffer._it_sum.sum()))
            assert(self._replay_buffer._it_sum.sum() > 0)
            assert(not np.isnan(self._replay_buffer._it_min.min()))
            assert(not np.isinf(self._replay_buffer._it_min.min()))
            assert(self._replay_buffer._it_min.min() > 0)

            # Early stop
            if np.isnan(ve_loss) or np.isnan(ve_loss2):
                logger.error("Training: value estimator loss is nan, stop training")
                self._stop_training = True

            # Some basic summary of training loss
            if self._stats_tot_steps % self._policy_and_target_update_freq == 0:
                # Update the actor policy using the sampled gradient
                self._policy_estimator.update(current_state_batch, grads[0])
                self._policy_estimator.update_target_network()
                self._value_estimator1.update_target_network()
                self._value_estimator2.update_target_network()

    def _evaluate(self):
        pass
