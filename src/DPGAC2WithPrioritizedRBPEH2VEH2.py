"""
With prioritized replay buffer
"""
import argparse
import copy
import gym
import itertools
import numpy as np
import tensorflow as tf
import EnvironmentRunner
#from tensorflow.python import debug as tf_debug
from Estimators.DPGMultiPerceptronPolicyEstimator import DPGMultiPerceptronPolicyEstimator
from Estimators.DPGMultiPerceptronValueEstimator import DPGMultiPerceptronValueEstimator
from Agents.DPGAC2Agent import DPGAC2WithPrioritizedRB
from Utils import SummaryWriter
from Utils import OrnsteinUhlenbeckActionNoise
from Utils import PrioritizedReplayBuffer
from Utils import getModuleLogger
from EnvironmentRunner import runEnvironmentWithAgent

# Module logger
logger = getModuleLogger(__name__)

def MakeDPGAC2WithPrioritizedRBPEH2VEH2(session, env, args):
    # The number of states about the environment the agent can observe
    observation_space_dim = env.observation_space.shape[0]
    # The number of actions the agent can make
    action_space_dim = env.action_space.shape[0]
    assert(env.action_space.high[0] == -env.action_space.low[0])

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

    policy_estimator = DPGMultiPerceptronPolicyEstimator(
            sess=session,
            state_dim=observation_space_dim,
            action_dim=action_space_dim,
            h_layer_shapes=pe_layer_shapes,
            learning_rate=args.pe_learning_rate,
            action_bound=env.action_space.high[0],
            tau=args.tau,
            minibatch_size=2**args.minibatch_size_log
            )

    print("TO:!!!")
    print(policy_estimator.get_num_trainable_vars())
    value_estimator = DPGMultiPerceptronValueEstimator(
            sess=session,
            state_dim=observation_space_dim,
            action_dim=action_space_dim,
            h_layer_shapes=ve_layer_shapes,
            learning_rate=args.ve_learning_rate,
            tau=args.tau,
            num_actor_vars=policy_estimator.get_num_trainable_vars()
            )

    summary_writer = SummaryWriter(session, args.summary_dir,[
        "TotalReward",
        "AverageMaxQ",
        "BestAverage",
        #"ValueEstimatorTrainLoss"
        ])

    estimator_saver_recent = tf.train.Saver(max_to_keep=args.max_estimators_to_keep)
    estimator_saver_best = tf.train.Saver(max_to_keep=1)

    # TODO: remove hardcoded value
    alpha = 0.3
    replay_buffer = PrioritizedReplayBuffer(10 ** args.replay_buffer_size_log, alpha)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_dim))

    return DPGAC2WithPrioritizedRB(
                sess=session,
                policy_estimator=policy_estimator,
                value_estimator=value_estimator,
                replay_buffer=replay_buffer,
                discount_factor=args.discount_factor,
                num_episodes=args.num_episodes,
                max_episode_length=args.max_episode_length,
                minibatch_size=2**args.minibatch_size_log,
                actor_noise=actor_noise,
                summary_writer=summary_writer,
                estimator_save_dir=args.estimator_dir,
                estimator_saver_recent=estimator_saver_recent,
                estimator_saver_best=estimator_saver_best,
                recent_save_freq=args.estimator_save_freq,
                replay_buffer_save_dir=args.replay_buffer_save_dir,
                replay_buffer_save_freq=args.replay_buffer_save_freq,
                num_updates=args.num_updates,
                )

def getArgParser():
    # Build parser
    parser = EnvironmentRunner.getArgParser()
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

    return parser

if __name__ == "__main__":

    args = getArgParser().parse_args()
    args.agent_name="DPGAC2WithPrioritizedRBPEH2VEH2"

    logger.info("Starting Agent {} in Environment {}".format(args.agent_name, args.env_name))
    runEnvironmentWithAgent(args)
    logger.info("Exiting")
