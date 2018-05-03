import numpy as np
import random
import tensorflow as tf

from Utils import ReplayBuffer
from Utils import getModuleLogger
from Utils import generateRandomAction

# Module logger
logger = getModuleLogger(__name__)

class AgentBase(object):
    def __init__(self, sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise, summary_writer,
            estimator_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, num_updates=1):
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

        self._last_state = None
        self._last_action = None
        # Our objective metric
        self._best_average = None

        # To implement early stop
        self._stop_training = False
        self._last_average = None
        self._num_non_imp_eps = 0  # Maximal number of non-improving episodes
        self._max_num_non_imp_eps = 200  # Maximal number of non-improving episodes

        # Summary and checkpoints
        self._summary_writer = summary_writer
        self._estimator_dir = estimator_dir
        self._estimator_saver_recent = estimator_saver_recent
        self._estimator_saver_best = estimator_saver_best
        self._recent_save_freq = recent_save_freq
        self._replay_buffer_save_dir = replay_buffer_save_dir
        self._replay_buffer_save_freq = replay_buffer_save_freq

        # Number of updates per training step
        self._num_updates = num_updates

    def save(self, estimator_dir, is_best=False, step=None, write_meta_graph=False):
        if write_meta_graph:
            tf.train.export_meta_graph(filename="{}/{}.meta".format(estimator_dir, self.__class__.__name__))
            logger.info("{} meta graph saved".format(self.__class__.__name__))
        else:
            if is_best:
                self._estimator_saver_best.save(self._sess, "{}/best/{}".format(estimator_dir, self.__class__.__name__), global_step=step,
                    write_meta_graph=False)
            else:
                self._estimator_saver_recent.save(self._sess, "{}/recent/{}".format(estimator_dir, self.__class__.__name__), global_step=step,
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
        return self._best_average

    def act(self, observation, last_reward, termination, episode_start_num, episode_num, episode_num_var, is_learning=False):
        self._total_reward += last_reward
        self._step += 1
        # Putting a limit on how long the a single trial can run for simplicity
        if self._step > self._max_episode_length:
            termination = True

        current_state = observation.reshape(1, -1)

        # Initialize the last state and action
        if self._last_state is None:
            self._last_state = current_state
            best_action = self._policy_estimator.predict(self._last_state)
            # Add exploration noise when training
            if is_learning:
                best_action += self._actor_noise()
            self._last_action = best_action
            return best_action, termination

        # Store the last step
        self._replay_buffer.add(self._last_state.squeeze().copy(), self._last_action.squeeze().copy(), last_reward, termination,
                current_state.squeeze().copy())

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
            self._rewards_list.append(self._total_reward)
            average = np.mean(self._rewards_list[-self._num_rewards_to_average:])

            # Update episode number variable
            self._sess.run(tf.assign(episode_num_var, episode_num))

            # Check for improvements
            if len(self._rewards_list) >= self._num_rewards_to_average and\
                (self._best_average is None or self._best_average < average):
                self._best_average = average
                improve_str = '*'
            else:
                improve_str = ''

            # Log the episode summary
            log_string = "Episode {0:>5}/{1:>5} ({2:>5}/{3:>5} in this run), " +\
                         "R {4:>9.3f}, Ave R {5:>9.3f} {7}, Ave Max Q {6:>9.3f}"

            logger.info(log_string.format(episode_num, self._num_episodes + episode_start_num - 1,
                episode_num - episode_start_num + 1,
                self._num_episodes,
                self._total_reward[0], average, self._max_q / float(self._step), improve_str))

            # Checkpoint
            if is_learning:
                if improve_str == '*':
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_dir, is_best=True, step=episode_num, write_meta_graph=False)
                if (episode_num % self._recent_save_freq == 0 or episode_num >= self._num_episodes +\
                    episode_start_num - 1):
                    logger.info("Saving agent checkpoints")
                    self.save(self._estimator_dir, step=episode_num, write_meta_graph=False)
                # Save summary
                self._summary_writer.writeSummary({
                    "TotalReward": self._total_reward[0],
                    "AverageMaxQ": self._max_q / float(self._step),
                    "BestAverage": self._best_average,
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
            self._total_reward = 0.0
            self._max_q = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None

        return best_action, termination

class DPGAC2Agent(AgentBase):
    def __init__(self, sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise, summary_writer,
            estimator_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, num_updates=1):
         super().__init__(sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise, summary_writer,
            estimator_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, num_updates)

    def _train(self):
        max_qs = []
        for _ in range(self._num_updates):
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

            max_qs.append(np.amax(predicted_q_value))

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

        self._max_q += max(max_qs)

class DPGAC2WithDemoAgent(AgentBase):
    def __init__(self, sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise,
            summary_writer, imitation_summary_writer,
            estimator_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, num_updates=1):
         super().__init__(sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise, summary_writer,
            estimator_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, num_updates)
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
            for exp in self._replay_buffer.sample_batch_full(self._minibatch_size):
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
            for exp in self._replay_buffer.sample_batch_full(self._minibatch_size):
                current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch = exp
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

                max_qs.append(np.amax(predicted_q_value))

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
        self._total_reward += last_reward
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
        self._replay_buffer.add(self._last_state.squeeze().copy(), self._last_action.squeeze().copy(), last_reward, termination,
                current_state.squeeze().copy())

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
            self._rewards_list.append(self._total_reward)
            average = np.mean(self._rewards_list[-self._num_rewards_to_average:])

            # Update episode number variable
            self._sess.run(tf.assign(episode_num_var, episode_num))

            # Check for improvements
            if len(self._rewards_list) >= self._num_rewards_to_average and\
                (self._best_average is None or self._best_average < average):
                self._best_average = average
                improve_str = '*'
            else:
                improve_str = ''

            # Log the episode summary
            log_string = "Episode {0:>5}/{1:>5} ({2:>5}/{3:>5} in this run), " +\
                         "R {4:>9.3f}, Ave R {5:>9.3f} {7}, Ave Max Q {6:>9.3f}"

            logger.info(log_string.format(episode_num, self._num_episodes + episode_start_num - 1,
                episode_num - episode_start_num + 1,
                self._num_episodes,
                self._total_reward[0], average, self._max_q / float(self._step), improve_str))

            # Checkpoint
            if is_learning:
                if improve_str == '*':
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_dir, is_best=True, step=episode_num, write_meta_graph=False)
                if (episode_num % self._recent_save_freq == 0 or episode_num >= self._num_episodes +\
                    episode_start_num - 1):
                    logger.info("Saving agent checkpoints")
                    self.save(self._estimator_dir, step=episode_num, write_meta_graph=False)
                # Save summary
                self._summary_writer.writeSummary({
                    "TotalReward": self._total_reward[0],
                    "AverageMaxQ": self._max_q / float(self._step),
                    "BestAverage": self._best_average,
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
            self._total_reward = 0.0
            self._max_q = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None

        return best_action, termination

    def _train(self):
        max_qs = []
        for _ in range(self._num_updates):
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

            max_qs.append(np.amax(predicted_q_value))

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

        self._max_q += max(max_qs)

class DPGAC2WithMultiPModelAndDemoAgent(DPGAC2WithDemoAgent):
    def __init__(self, sess, policy_estimator, value_estimator, model_estimator, model_eval_replay_buffer,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise,
            summary_writer, imitation_summary_writer, model_summary_writer,
            estimator_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, num_updates=1):
         super().__init__(sess, policy_estimator, value_estimator,
            discount_factor, num_episodes, max_episode_length, minibatch_size, replay_buffer_size, actor_noise,
            summary_writer, imitation_summary_writer,
            estimator_dir, estimator_saver_recent, estimator_saver_best, recent_save_freq, replay_buffer_save_dir,
            replay_buffer_save_freq, num_updates)
         self._model_summary_writer = model_summary_writer
         self._model_estimator = model_estimator
         self._model_eval_replay_buffer = model_eval_replay_buffer
         self._best_action_from_model = True
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
            for exp in self._replay_buffer.sample_batch_full(self._minibatch_size):
                current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch = exp
                model_loss = self._model_estimator.update(current_state_batch, action_batch, next_state_batch - current_state_batch)
                self._model_summary_writer.writeSummary({
                    "ModelTrainLoss": model_loss
                    }, self._model_train_step)
            self._model_train_step += 1

    def _trainModel(self, steps):
        for i in range(steps):
            exp = self._replay_buffer.sample_batch(self._minibatch_size)
            current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch = exp
            _, _, model_loss = self._model_estimator.update(current_state_batch, action_batch, next_state_batch - current_state_batch)
            if i % 100 == 0:
                eval_exp = self._model_eval_replay_buffer.sample_batch(self._model_eval_replay_buffer.size())
                current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch = eval_exp
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
            for exp in self._replay_buffer.sample_batch_full(self._minibatch_size):
                current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch = exp
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

                max_qs.append(np.amax(predicted_q_value))

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
        self._total_reward += last_reward
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
        self._replay_buffer.add(self._last_state.squeeze().copy(), self._last_action.squeeze().copy(), last_reward, termination,
                current_state.squeeze().copy())

        self._last_state = current_state.copy()

        if is_learning and self._replay_buffer.size() >= self._minibatch_size and not self._stop_training:
            self._train()

        best_action = self._getBestAction()

        self._last_action = best_action

        if termination:
            # Switch between getting action from model and policy, but get from policy exclusively after episode
            # _stop_best_action_from_model_episode
            # TODO:
            if episode_num >= self._stop_best_action_from_model_episode:
                self._best_action_from_model = False
            else:
                self._best_action_from_model = not self._best_action_from_model
            #self._best_action_from_model = True

            # Record cumulative reward of trial
            self._rewards_list.append(self._total_reward)
            average = np.mean(self._rewards_list[-self._num_rewards_to_average:])

            # Update episode number variable
            self._sess.run(tf.assign(episode_num_var, episode_num))

            # Check for improvements
            if len(self._rewards_list) >= self._num_rewards_to_average and\
                (self._best_average is None or self._best_average < average):
                self._best_average = average
                improve_str = '*'
            else:
                improve_str = ''

            # Log the episode summary
            log_string = "Episode {0:>5}/{1:>5} ({2:>5}/{3:>5} in this run), " +\
                         "R {4:>9.3f}, Ave R {5:>9.3f} {7}, Ave Max Q {6:>9.3f}"

            logger.info(log_string.format(episode_num, self._num_episodes + episode_start_num - 1,
                episode_num - episode_start_num + 1,
                self._num_episodes,
                self._total_reward[0], average, self._max_q / float(self._step), improve_str))

            # Checkpoint
            if is_learning:
                if improve_str == '*':
                    logger.info("Saving best agent so far")
                    self.save(self._estimator_dir, is_best=True, step=episode_num, write_meta_graph=False)
                if (episode_num % self._recent_save_freq == 0 or episode_num >= self._num_episodes +\
                    episode_start_num - 1):
                    logger.info("Saving agent checkpoints")
                    self.save(self._estimator_dir, step=episode_num, write_meta_graph=False)
                # Save summary
                self._summary_writer.writeSummary({
                    "TotalReward": self._total_reward[0],
                    "AverageMaxQ": self._max_q / float(self._step),
                    "BestAverage": self._best_average,
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
            self._total_reward = 0.0
            self._max_q = 0.0
            self._step = 0
            self._last_state = None
            self._last_action = None

        return best_action, termination

    def _train(self):
        max_qs = []
        for _ in range(self._num_updates):
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

            max_qs.append(np.amax(predicted_q_value))

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

        self._max_q += max(max_qs)
