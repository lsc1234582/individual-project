import argparse
import gym
import itertools
import numpy as np
import tensorflow as tf

#from tensorflow.python import debug as tf_debug
from Estimators.SPGMultiPerceptronPolicyEstimator import SPGMultiPerceptronPolicyEstimator
from Estimators.SPGMultiPerceptronValueEstimator import SPGMultiPerceptronValueEstimator
from Agents.SPGAC2Agent import SPGAC2Agent
from Utils import SummaryWriter
from Utils import getModuleLogger
from EnvironmentRunner import runEnvironmentWithAgent

# Module logger
logger = getModuleLogger(__name__)

def MakeSPGAC2PEH1VEH1(session, env, args):
    # The number of states about the environment the agent can observe
    observation_space_dim = env.observation_space.shape[0]
    # The number of actions the agent can make
    action_space_dim = env.action_space.shape[0]

    # The shapes of the layers of the policy estimator
    pe_layer_shapes = [
            observation_space_dim,
            max(1, int(args.pe_h1_multiplier * observation_space_dim)),
            action_space_dim
            ]
    # The shapes of the layers of the value estimator
    ve_layer_shapes = [
            observation_space_dim,
            max(1, int(args.ve_h1_multiplier * observation_space_dim)),
            action_space_dim
            ]


    policy_estimator = SPGMultiPerceptronPolicyEstimator(
            session,
            pe_layer_shapes,
            learning_rate=args.pe_learning_rate,
            action_lo=env.action_space.low[0],
            action_hi=env.action_space.high[0]
            )

    value_estimator = SPGMultiPerceptronValueEstimator(
            session,
            ve_layer_shapes,
            learning_rate=args.ve_learning_rate,
            )

    summary_writer = SummaryWriter(session, args.summary_dir,[
        "TotalReward",
        "PolicyEstimatorTrainLoss",
        "ValueEstimatorTrainLoss"
        ])

    return SPGAC2Agent(
                sess=session,
                policy_estimator=policy_estimator,
                value_estimator=value_estimator,
                discount_factor=args.discount_factor,
                num_episodes=args.num_episodes,
                max_episode_length=args.max_episode_length,
                minibatch_size=2**args.minibatch_size_log,
                replay_buffer_size=10**args.replay_buffer_size_log,
                summary_writer=summary_writer
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="provide arguments for SPGAC2PEH1VEH1 agent")

    # Agent and run parameters
    parser.add_argument("--is-agent-learning", help="Is Agent learning", action="store_true")
    parser.add_argument("--num-episodes", help="max num of episodes to do while training", default=500)
    parser.add_argument("--max-episode-length", help="max length of 1 episode", default=200)
    parser.add_argument("--pe-learning-rate", help="Policy esitmator learning rate", default=0.0001)
    parser.add_argument("--pe-h1-multiplier", help="Policy estimator hidden layer 1 size multiplier", default=10)
    parser.add_argument("--ve-learning-rate", help="Value esitmator learning rate", default=0.01)
    parser.add_argument("--ve-h1-multiplier", help="Value estimator hidden layer 1 size multiplier", default=10)
    parser.add_argument("--discount-factor", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--minibatch-size-log", help="size of minibatch for minibatch-SGD as exponent of 2", default=7)
    parser.add_argument("--replay-buffer-size-log", help="max size of the replay buffer as exponent of 10", default=5)

    # Environment parameters
    parser.add_argument("--env-name", help="choose the env[VREPPushTask, Pendulum-v0]", required=True)
    parser.add_argument("--random-seed", help="random seed for repeatability", default=1234)
    parser.add_argument("--render-env", help="render the env", action="store_true")

    # Other parameters
    parser.add_argument("--summary-dir", help="directory for storing tensorboard info", required=True)

    parser.set_defaults(is_agent_learning=True)
    parser.set_defaults(render_env=False)

    args = parser.parse_args()

    args.agent_name = "SPGAC2PEH1VEH1"
    logger.info("Starting Agent in Environment {}".format(args.agent_name, args.env_name))
    best_score = runEnvironmentWithAgent(MakeSPGAC2PEH1VEH1, args)
    logger.info("Exiting")
