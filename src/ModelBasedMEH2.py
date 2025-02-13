import argparse
import copy
import gym
import itertools
import numpy as np
import tensorflow as tf
import EnvironmentRunner
#from tensorflow.python import debug as tf_debug
from Estimators.MultiPerceptronModelEstimator import MultiPerceptronModelEstimator
from Agents.ModelBasedAgent import ModelBasedAgent
from Utils.Utils import ReplayBuffer
from Utils.Utils import SummaryWriter
from Utils.Utils import ReplayBuffer
from Utils.Utils import OrnsteinUhlenbeckActionNoise
from Utils.Utils import getModuleLogger
from EnvironmentRunner import runEnvironmentWithAgent
from Utils.MPIRunningMeanStd import RunningMeanStd

# Module logger
logger = getModuleLogger(__name__)

def MakeModelBasedMEH2(session, env, args):
    # The number of states about the environment the agent can observe
    observation_space_dim = env.observation_space.shape[0]
    # The number of actions the agent can make
    action_space_dim = env.action_space.shape[0]
    assert(env.action_space.high[0] == -env.action_space.low[0])

    # TODO: remove hardcoded value
    state_range = [-999, 999]
    return_range = [-999, 999]
    # State normalization.
    if args.normalize_states:
        with tf.variable_scope('state_rms'):
            state_rms = RunningMeanStd(shape=env.observation_space.shape)
        with tf.variable_scope('state_change_rms'):
            state_change_rms = RunningMeanStd(shape=env.observation_space.shape)
    else:
        state_rms = None
        state_change_rms = None

    # Return normalization.
    if args.normalize_returns:
        with tf.variable_scope('return_rms'):
            return_rms = RunningMeanStd()
    else:
        return_rms = None

    me_layer_shapes = [
            max(1, int(args.me_h1_multiplier * observation_space_dim)),
            max(1, int(args.me_h2_multiplier * observation_space_dim)),
            ]

    model_estimator = MultiPerceptronModelEstimator(
            sess=session,
            state_rms=state_rms,
            state_change_rms=state_change_rms,
            state_range=state_range,
            state_dim=observation_space_dim,
            action_dim=action_space_dim,
            h_layer_shapes=me_layer_shapes,
            learning_rate=args.me_learning_rate
            )

    summary_writer = SummaryWriter(session, args.summary_dir) if args.log_stats_freq > 0 else None

    estimator_saver_recent = tf.train.Saver(max_to_keep=args.max_estimators_to_keep)
    estimator_saver_best = tf.train.Saver(max_to_keep=1)

    replay_buffer = ReplayBuffer(10 ** args.replay_buffer_size_log)
    if args.eval_replay_buffer_load_dir is not None:
        eval_replay_buffer = ReplayBuffer(10 ** args.replay_buffer_size_log)
    else:
        eval_replay_buffer = None

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_dim))

    return ModelBasedAgent(
                sess=session,
                env=env,
                model_estimator=model_estimator,
                replay_buffer=replay_buffer,
                discount_factor=args.discount_factor,
                num_train_steps=args.num_train_steps,
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
                normalize_states=args.normalize_states,
                state_rms=state_rms,
                state_change_rms=state_change_rms,
                normalize_returns=args.normalize_returns,
                return_rms=return_rms,
                num_updates=args.num_updates,
                log_stats_freq=args.log_stats_freq,
                train_freq=args.train_freq,
                eval_replay_buffer=eval_replay_buffer,
                num_test_eps=args.num_test_eps,
                test_freq=args.test_freq,
                horizon_length=args.mpc_horizon_length,
                num_random_action=args.mpc_num_action,
                )

def getArgParser():
    # Build argument parser
    parser = EnvironmentRunner.getArgParser()
    # Add Agent parameters
    parser.add_argument("--discount-factor", help="discount factor for critic updates", type=float, default=0.99)
    parser.add_argument("--minibatch-size-log", help="size of minibatch for minibatch-SGD as exponent of 2",
            type=int, default=7)
    parser.add_argument("--replay-buffer-size-log", help="max size of the replay buffer as exponent of 10", type=int, default=6)
    parser.add_argument("--me-learning-rate", help="Model esitmator learning rate", type=float, default=0.001)
    parser.add_argument("--me-h1-multiplier", help="Model estimator hidden layer 1 size multiplier", type=float, default=10)
    parser.add_argument("--me-h2-multiplier", help="Model estimator hidden layer 2 size multiplier", type=float, default=10)
    parser.add_argument("--mpc-horizon-length", help="MPC horizon length", type=int, default=5)
    parser.add_argument("--mpc-num-action", help="MPC number of actions", type=int, default=50)

    return parser

if __name__ == "__main__":
    import json

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    args = getArgParser().parse_args()

    if args.config_json is not None:
        with open(args.config_json, "r") as f:
            config_dicts = json.load(f)
        args_dicts = vars(args)
        args_dicts.update(config_dicts)
        args = Args(**args_dicts)

    args.agent_name = "ModelBasedMEH2"

    logger.info("Starting Agent {} in Environment {}".format(args.agent_name, args.env_name))
    runEnvironmentWithAgent(args)
    logger.info("Exiting")
