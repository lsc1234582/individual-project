import numpy as np
import tensorflow as tf
from Utils.Utils import getModuleLogger
from Utils.Utils import generateRandomAction
from Agents.AgentBase import AgentBase

# Module logger
logger = getModuleLogger(__name__)

class ModelBasedAgent(AgentBase):
    def __init__(self, sess, env, model_estimator, replay_buffer,
            discount_factor, num_train_steps, max_episode_length, minibatch_size, actor_noise, summary_writer,
            estimator_save_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, normalize_states, state_rms, state_change_rms, normalize_returns, return_rms, num_updates=1,
            log_stats_freq=1, train_freq=1, eval_replay_buffer=None, test_freq=50, num_test_eps=20, horizon_length=5,
            num_random_action=50):
        super().__init__(sess, env, replay_buffer,
            discount_factor, num_train_steps, max_episode_length, minibatch_size, actor_noise, summary_writer,
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
        # Planning horizon for model-based learning
        self._horizon_length = horizon_length
        # Number of random actions to take
        self._num_random_action = num_random_action
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

    def _getBestActionAux(self, num_rollout, batch_size, state):
        current_state = np.concatenate([np.copy(state) for _ in range(num_rollout)], axis=0)
        next_state = np.copy(current_state)

        num_rollout *= batch_size
        assert current_state.shape[0] == num_rollout and current_state.shape[1] ==\
                    self._model_estimator._state_dim

        best_action = None
        first_actions = None
        horizon_reward = np.zeros((num_rollout, 1))
        done_selection = np.array([False for _ in range(num_rollout)])
        for j in range(self._horizon_length):
            if num_rollout <= 0:
                break
            actions = generateRandomAction(self._env.action_space.high[0], self._env.action_space.shape[0] *
                    num_rollout).reshape(num_rollout, -1)
            if j == 0:
                first_actions = actions
            # Proceed to next state
            assert current_state.shape[0] == num_rollout and current_state.shape[1] == self._model_estimator._state_dim
            next_state[~done_selection]= current_state + self._model_estimator.predict(current_state, actions)
            horizon_reward[~done_selection] += self._env.getRewards(current_state, actions, next_state[~done_selection])
            # Detect termination, only proceed with non-terminated states
            done_selection = np.array(self._env._reachedGoalState(next_state)).flatten()
            current_state = np.copy(next_state[~done_selection, :])
            num_rollout = current_state.shape[0]

        assert first_actions is not None
        return horizon_reward, first_actions

    def _getBestAction(self, state):
        """
        Assume reward function is given.
        NOTE: Doesn't support batch.

        Args
        -------
        state:          array(state_dim)/array(1, state_dim)

        Returns
        best_action:    array(-1, action_dim)
        -------
        """
        assert self._num_random_action != 0
        assert self._horizon_length != 0
        state = state.reshape(-1, self._model_estimator._state_dim)
        batch_size = state.shape[0]

        num_rollout = self._num_random_action

        horizon_reward, first_actions = self._getBestActionAux(num_rollout, batch_size, state)

        # Simple version to improve performance during acting
        if batch_size == 1:
            max_horizon_reward = np.max(horizon_reward)
            best_action_selection = (horizon_reward == max_horizon_reward).squeeze()
            best_actions = first_actions[best_action_selection][0]
        else:
            horizon_reward = horizon_reward.reshape(self._num_random_action, batch_size)
            max_horizon_reward = np.max(horizon_reward, axis=0)
            best_action_selection = horizon_reward == max_horizon_reward
            # Remove duplicate best actions within all batches
            mask = np.array([False for i in range(batch_size)])
            for i in range(self._num_random_action):
                best_action_selection[i] &= ~mask
                mask |= best_action_selection[i]
            best_actions = first_actions.reshape(num_rollout, batch_size,
                    self._model_estimator._action_dim)[best_action_selection, :]

        return best_actions.reshape(batch_size, self._model_estimator._action_dim)


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

