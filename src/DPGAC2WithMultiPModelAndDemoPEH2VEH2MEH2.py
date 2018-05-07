import argparse
import collections
import random
from EnvironmentFactory import EnvironmentContext
from Utils import SortedDisplayDict
import copy
import gym
import itertools
import numpy as np
import os
import tensorflow as tf
import sklearn
#from tensorflow.python import debug as tf_debug
from Estimators.DPGMultiPerceptronPolicyEstimator import DPGMultiPerceptronPolicyEstimator
from Estimators.DPGMultiPerceptronValueEstimator import DPGMultiPerceptronValueEstimator
from Estimators.MultiPerceptronModelEstimator import MultiPerceptronModelEstimator
from Agents.DPGAC2Agent import DPGAC2WithMultiPModelAndDemoAgent
from Utils import ReplayBuffer
from Utils import SummaryWriter
from Utils import ReplayBuffer
from Utils import OrnsteinUhlenbeckActionNoise
from Utils import getModuleLogger

# Module logger
logger = getModuleLogger(__name__)

def MakeDPGAC2WithMultiPModelAndDemoPEH2VEH2MEH2(session, env, args):
    # The number of states about the environment the agent can observe
    observation_space_dim = env.observation_space.shape[0]
    # The number of actions the agent can make
    action_space_dim = env.action_space.shape[0]
    assert(env.action_space.high[0] == -env.action_space.low[0])
    assert(not args.replay_buffer_random_load_dir is None)

    # The shapes of the hidden layers of the policy estimator
    pe_layer_shapes = [
            max(1, int(args.pe_h1_multiplier * observation_space_dim)),
            max(1, int(args.pe_h2_multiplier * observation_space_dim)),
            ]

    # This implementation assumes a particular architecture for value estimator
    # The shapes of the hidden layers of the policy estimator
    ve_layer_shapes = [
            max(1, int(args.ve_h1_multiplier * observation_space_dim)),
            max(1, int(args.ve_h2_multiplier * observation_space_dim)),
            ]

    me_layer_shapes = [
            max(1, int(args.me_h1_multiplier * observation_space_dim)),
            max(1, int(args.me_h2_multiplier * observation_space_dim)),
            ]

    replay_buffer_random = ReplayBuffer(1)
    replay_buffer_random.load(args.replay_buffer_random_load_dir)
    logger.info("Replay buffer random loaded.")
    state, _, _, _, _ = replay_buffer_random.sample_batch(replay_buffer_random.size())

    state_processor = sklearn.preprocessing.StandardScaler()
    state_processor.fit(state)
    logger.info("State processor fitted.")

    policy_estimator = DPGMultiPerceptronPolicyEstimator(
            sess=session,
            state_dim=observation_space_dim,
            action_dim=action_space_dim,
            h_layer_shapes=pe_layer_shapes,
            learning_rate=args.pe_learning_rate,
            action_bound=env.action_space.high[0],
            tau=args.tau,
            minibatch_size=2**args.minibatch_size_log,
            imitation_learning_rate=args.pe_imitation_learning_rate,
            imitation_minibatch_size=2**args.pe_imitation_minibatch_size_log,
            )

    value_estimator = DPGMultiPerceptronValueEstimator(
            sess=session,
            state_dim=observation_space_dim,
            action_dim=action_space_dim,
            h_layer_shapes=ve_layer_shapes,
            learning_rate=args.ve_learning_rate,
            tau=args.tau,
            num_actor_vars=policy_estimator.get_num_trainable_vars()
            )

    model_estimator = MultiPerceptronModelEstimator(
            sess=session,
            state_processor=state_processor,
            state_dim=observation_space_dim,
            action_dim=action_space_dim,
            h_layer_shapes=me_layer_shapes,
            learning_rate=args.me_learning_rate
            )

    summary_writer = SummaryWriter(session, args.summary_dir,[
        "TotalReward",
        "AverageMaxQ",
        "BestAverage",
        #"ValueEstimatorTrainLoss"
        ])

    imitation_summary_writer = SummaryWriter(session, os.path.join(args.summary_dir, "ImitationLearning"),[
        "TrainLoss",
        ])

    model_summary_writer = SummaryWriter(session, os.path.join(args.summary_dir, "ModelLearning"),[
        "ModelTrainLoss",
        "ModelEvaluationLoss",
        ])

    estimator_saver_recent = tf.train.Saver(max_to_keep=args.max_estimators_to_keep)
    estimator_saver_best = tf.train.Saver(max_to_keep=1)
    replay_buffer = ReplayBuffer(10 ** args.replay_buffer_size_log)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_dim))

    return DPGAC2WithMultiPModelAndDemoAgent(
                sess=session,
                policy_estimator=policy_estimator,
                value_estimator=value_estimator,
                model_estimator=model_estimator,
                replay_buffer=replay_buffer,
                model_eval_replay_buffer=replay_buffer_random,
                discount_factor=args.discount_factor,
                num_episodes=args.num_episodes,
                max_episode_length=args.max_episode_length,
                minibatch_size=2**args.minibatch_size_log,
                actor_noise=actor_noise,
                summary_writer=summary_writer,
                imitation_summary_writer=imitation_summary_writer,
                model_summary_writer=model_summary_writer,
                estimator_save_dir=args.estimator_dir,
                estimator_saver_recent=estimator_saver_recent,
                estimator_saver_best=estimator_saver_best,
                recent_save_freq=args.estimator_save_freq,
                replay_buffer_save_dir=args.replay_buffer_save_dir,
                replay_buffer_save_freq=args.replay_buffer_save_freq,
                num_updates=args.num_updates,
                )

def runEnvironmentWithAgent(args):
    # dirty but works
    from AgentFactory import MakeAgent
    logger.info("Run info:")
    logger.info(SortedDisplayDict(vars(args)))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    # Set graph-level random seed to ensure repeatability of experiments
    tf.set_random_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    logger.info("Making environment {}".format(args.env_name))
    with EnvironmentContext(args.env_name) as env, tf.Session(config=config) as session:
        # To record progress across different training sessions
        global_episode_num = tf.Variable(0, name="global_episode_num", trainable=False)
        logger.info("Making agent {}".format(args.agent_name))
        agent = MakeAgent(session, env, args)

        session.run(tf.global_variables_initializer())
        if args.new_estimator:
            logger.info("Saving session meta file to {}".format(args.estimator_dir))
            agent.save(args.estimator_dir, step=0, write_meta_graph=True)
        else:
            logger.info("Restoring agent from {}".format(args.estimator_dir))
            agent.load(args.estimator_dir, is_best=(args.estimator_load_mode==1))
        if not args.replay_buffer_load_dir is None:
            logger.info("Restoring replay buffer from {}".format(args.replay_buffer_load_dir))
            agent.loadReplayBuffer(args.replay_buffer_load_dir)


        episode_start = session.run(global_episode_num) + 1
        logger.info("Continueing at episode {}".format(episode_start))
        # Run the environment feedback loop
        for episode_num in range(episode_start, episode_start + args.num_episodes):
            observation = env.reset()
            reward = 0.0
            done = False
            action, done = agent.act(observation, reward, done, episode_start, episode_num, global_episode_num,
                    is_learning=(not args.stop_agent_learning))

            while not done:
                if episode_num - episode_start <= 10 or episode_num >= int((2 * episode_start + args.num_episodes)/2):
                    if args.render_env:
                        env.render()
                    observation, reward, done, _ = env.step(action)
                else:
                    reward = np.array([env.__class__.getRewards(observation, action)])
                    observation += agent._model_estimator.predict(observation.reshape(1, -1), action.reshape(1,
                        -1)).squeeze()
                action, done = agent.act(observation, reward, done, episode_start, episode_num, global_episode_num,
                                         is_learning=(not args.stop_agent_learning))
                #logger.debug("Observation")
                #logger.debug(observation)
                #logger.debug("Action")
                #logger.debug(action)
                #logger.debug("Reward")
                #logger.debug(reward)
                #logger.debug("Done")
                #logger.debug(done)
            # No need to push forward when the agent stops training and has collected enough episodes to obtain a score
            if agent._stop_training and agent.score():
                logger.warn("Agent stopped training. Exiting experiment...")
                break

    logger.info("Best score: {}".format(agent.score()))
    logger.info("Exiting environment: {}".format(args.env_name))
    return agent.score()

def getArgParser():
    # Build argument parser
    parser = argparse.ArgumentParser(description="provide arguments for DPGAC2WithMultiPModelAndDemoPEH1VEH1MEH2 agent")

    # Session parameters
    parser.add_argument("--env-name", help="choose the env[VREPPushTask, Pendulum-v0]", required=True)
    parser.add_argument("--estimator-dir", help="directory for loading/storing estimators", required=True)
    parser.add_argument("--summary-dir", help="directory for storing tensorboard info", required=True)
    #parser.add_argument("--agent-name", help="name of the agent")
    parser.add_argument("--stop-agent-learning", help="Is Agent learning", action="store_true")
    parser.add_argument("--num-episodes", help="max num of episodes to do while training", type=int, default=500)
    parser.add_argument("--max-episode-length", help="max length of 1 episode", type=int, default=100)
    parser.add_argument("--random-seed", help="random seed for repeatability", type=int, default=1234)
    parser.add_argument("--render-env", help="render the env", action="store_true")
    parser.add_argument("--new-estimator", help="if creating new estimators instead of loading old ones", action="store_true")
    parser.add_argument("--max-estimators-to-keep", help="maximal number of estimators to keep checkpointing",
            type=int, default=2)
    parser.add_argument("--estimator-save-freq", help="estimator save frequency (per number of episodes)",
            type=int, default=50)
    parser.add_argument("--estimator-load-mode", help="0: load most recent 1: load best", type=int, default=0)
    parser.add_argument("--replay-buffer-load-dir", help="directory for loading replay buffer")
    parser.add_argument("--replay-buffer-save-dir", help="directory for storing replay buffer")
    parser.add_argument("--replay-buffer-save-freq", help="replay buffer save frequency (per number of episodes", type=int,
            default=500)

    parser.set_defaults(stop_agent_learning=False)
    parser.set_defaults(render_env=False)
    parser.set_defaults(new_estimator=False)

    # Add Agent parameters
    parser.add_argument("--pe-learning-rate", help="Policy esitmator learning rate", type=float, default=0.0001)
    parser.add_argument("--ve-learning-rate", help="Value esitmator learning rate", type=float, default=0.001)
    parser.add_argument("--pe-h1-multiplier", help="Policy estimator hidden layer 1 size multiplier", type=float, default=10)
    parser.add_argument("--ve-h1-multiplier", help="Value estimator hidden layer 1 size multiplier", type=float, default=10)
    parser.add_argument("--pe-h2-multiplier", help="Policy estimator hidden layer 2 size multiplier", type=float, default=10)
    parser.add_argument("--ve-h2-multiplier", help="Value estimator hidden layer 2 size multiplier", type=float, default=10)
    parser.add_argument("--discount-factor", help="discount factor for critic updates", type=float, default=0.99)
    parser.add_argument("--tau", help="soft target update parameter", type=float, default=0.001)
    parser.add_argument("--minibatch-size-log", help="size of minibatch for minibatch-SGD as exponent of 2",
            type=int, default=7)
    parser.add_argument("--replay-buffer-size-log", help="max size of the replay buffer as exponent of 10", type=int, default=6)
    parser.add_argument("--num-updates", help="number of estimator updates per training step", type=int, default=1)
    parser.add_argument("--pe-imitation-learning-rate", help="Policy esitmator imitation learning rate", type=float, default=0.001)
    parser.add_argument("--pe-imitation-minibatch-size-log", help="size of minibatch for imitation minibatch-SGD as exponent of 2",
            type=int, default=7)
    parser.add_argument("--me-learning-rate", help="Model esitmator learning rate", type=float, default=0.001)
    parser.add_argument("--me-h1-multiplier", help="Model estimator hidden layer 1 size multiplier", type=float, default=10)
    parser.add_argument("--me-h2-multiplier", help="Model estimator hidden layer 2 size multiplier", type=float, default=10)
    parser.add_argument("--replay-buffer-random-load-dir", help="directory for loading replay buffer random")

    return parser

if __name__ == "__main__":

    args = getArgParser().parse_args()
    args.agent_name="DPGAC2WithMultiPModelAndDemoPEH2VEH2MEH2"

    logger.info("Starting Agent {} in Environment {}".format(args.agent_name, args.env_name))
    runEnvironmentWithAgent(args)
    logger.info("Exiting")
