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
from Agents.DPGAC2Agent import DPGAC2Agent
from Utils import SummaryWriter
from Utils import OrnsteinUhlenbeckActionNoise
from Utils import ReplayBuffer
from Utils import getModuleLogger
from EnvironmentRunner import runEnvironmentWithAgent
from mpi_running_mean_std import RunningMeanStd

# Module logger
logger = getModuleLogger(__name__)

def MakeDPGAC2PEH2VEH2(session, env, args):
    # The number of states about the environment the agent can observe
    observation_space_dim = env.observation_space.shape[0]
    # The number of actions the agent can make
    action_space_dim = env.action_space.shape[0]
    assert(env.action_space.high[0] == -env.action_space.low[0])

    # TODO: remove hardcoded value
    normalize_states = True
    normalize_returns = False
    state_range = [-999, 999]
    return_range = [-999, 999]
    # State normalization.
    if normalize_states:
        with tf.variable_scope('state_rms'):
            state_rms = RunningMeanStd(shape=env.observation_space.shape)
    else:
        state_rms = None

    # Return normalization.
    if normalize_returns:
        with tf.variable_scope('return_rms'):
            return_rms = RunningMeanStd()
    else:
        return_rms = None

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
            state_range=state_range,
            learning_rate=args.pe_learning_rate,
            action_bound=env.action_space.high[0],
            tau=args.tau,
            minibatch_size=2**args.minibatch_size_log,
            state_rms=state_rms
            )

    value_estimator = DPGMultiPerceptronValueEstimator(
            sess=session,
            state_dim=observation_space_dim,
            action_dim=action_space_dim,
            h_layer_shapes=ve_layer_shapes,
            state_range=state_range,
            return_range=return_range,
            learning_rate=args.ve_learning_rate,
            tau=args.tau,
            num_actor_vars=policy_estimator.get_num_trainable_vars(),
            state_rms=state_rms,
            return_rms=return_rms
            )

    summary_writer = SummaryWriter(session, args.summary_dir) if not args.stop_agent_learning else None

    estimator_saver_recent = tf.train.Saver(max_to_keep=args.max_estimators_to_keep)
    estimator_saver_best = tf.train.Saver(max_to_keep=1)

    replay_buffer = ReplayBuffer(10 ** args.replay_buffer_size_log, debug=True)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_dim))

    return DPGAC2Agent(
                sess=session,
                policy_estimator=policy_estimator,
                value_estimator=value_estimator,
                discount_factor=args.discount_factor,
                num_episodes=args.num_episodes,
                max_episode_length=args.max_episode_length,
                minibatch_size=2**args.minibatch_size_log,
                actor_noise=actor_noise,
                replay_buffer=replay_buffer,
                summary_writer=summary_writer,
                estimator_save_dir=args.estimator_dir,
                estimator_saver_recent=estimator_saver_recent,
                estimator_saver_best=estimator_saver_best,
                recent_save_freq=args.estimator_save_freq,
                replay_buffer_save_dir=args.replay_buffer_save_dir,
                replay_buffer_save_freq=args.replay_buffer_save_freq,
                normalize_states=normalize_states,
                state_rms=state_rms,
                normalize_returns=normalize_returns,
                return_rms=return_rms,
                num_updates=args.num_updates,
                log_stats_freq=args.log_stats_freq,
                train_freq=args.train_freq,
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

    return parser

if __name__ == "__main__":

    args = getArgParser().parse_args()
    args.agent_name="DPGAC2PEH2VEH2"

    logger.info("Starting Agent {} in Environment {}".format(args.agent_name, args.env_name))
    runEnvironmentWithAgent(args)
    logger.info("Exiting")
