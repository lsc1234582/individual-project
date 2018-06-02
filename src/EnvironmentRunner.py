import argparse
import collections
import numpy as np
import random
import tensorflow as tf
from EnvironmentFactory import EnvironmentContext
from Utils import getModuleLogger
from Utils import SortedDisplayDict

# Module logger
logger = getModuleLogger(__name__)


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
    with EnvironmentContext(args.env_name, port=args.env_vrep_port) as env, tf.Session(config=config) as session:
        # To record progress across different training sessions
        global_episode_num = tf.Variable(0, name="global_episode_num", trainable=False)
        logger.info("Making agent {}".format(args.agent_name))
        agent = MakeAgent(session, env, args)
        agent.initialize()

        if args.new_estimator:
            logger.info("Saving session meta file to {}".format(args.estimator_dir))
            agent.save(args.estimator_dir, step=0, write_meta_graph=True)
        else:
            logger.info("Restoring agent from {}".format(args.estimator_dir))
            agent.load(args.estimator_dir, is_best=(args.estimator_load_mode==1))
        if args.replay_buffer_load_dir is not None:
            logger.info("Restoring replay buffer from {}".format(args.replay_buffer_load_dir))
            agent.loadReplayBuffer(args.replay_buffer_load_dir)
        if args.eval_replay_buffer_load_dir is not None and agent._eval_replay_buffer is not None:
            logger.info("Restoring evaluation replay buffer from {}".format(args.eval_replay_buffer_load_dir))
            agent.loadEvalReplayBuffer(args.eval_replay_buffer_load_dir)


        episode_start = session.run(global_episode_num) + 1
        logger.info("Continueing at episode {}".format(episode_start))
        # Run the environment feedback loop
        train_episode_num = episode_start

        observation = env.reset()
        reward = np.array([0.0])
        done = False
        action, done, is_test_episode = agent.act(observation, reward, done, episode_start, train_episode_num, global_episode_num,
                is_learning=(not args.stop_agent_learning))
        for step in range(args.num_train_steps):
            if args.render_env:
                env.render()
            observation, reward, done, _ = env.step(action)
            action, done, is_test_episode = agent.act(observation, reward, done, episode_start, train_episode_num, global_episode_num,
                                     is_learning=(not args.stop_agent_learning))

            if done:
                observation = env.reset()
                reward = np.array([0.0])
                done = False
                action, done, is_test_episode = agent.act(observation, reward, done, episode_start, train_episode_num, global_episode_num,
                        is_learning=(not args.stop_agent_learning))
                if not is_test_episode:
                    train_episode_num += 1
                if agent._stop_training and agent.score():
                    # No need to push forward when the agent stops training and has collected enough episodes to obtain a score
                    logger.warn("Agent stopped training. Exiting experiment...")
                    break
                #logger.debug("Observation")
                #logger.debug(observation)
                #logger.debug("Action")
                #logger.debug(action)
                #logger.debug("Reward")
                #logger.debug(reward)
                #logger.debug("Done")
                #logger.debug(done)

    logger.info("Best score: {}".format(agent.score()))
    logger.info("Exiting environment: {}".format(args.env_name))
    return agent.score()

def getArgParser():
    # Build argument parser
    parser = argparse.ArgumentParser(description="provide arguments for DPGAC2PEH1VEH1 agent")

    # Session parameters
    parser.add_argument("--env-name", help="choose the env[VREPPushTask, Pendulum-v0]", required=True)
    parser.add_argument("--env-vrep-port", help="Port number to run vrep remote server", type=int, default=19997)
    parser.add_argument("--estimator-dir", help="directory for loading/storing estimators", required=True)
    parser.add_argument("--summary-dir", help="directory for storing stats (tensorboard info)", required=True)
    #parser.add_argument("--agent-name", help="name of the agent")
    parser.add_argument("--stop-agent-learning", help="Is Agent learning", action="store_true")
    parser.add_argument("--num-train-steps", help="max num of training steps", type=int, default=1000000)
    parser.add_argument("--max-episode-length", help="max length of 1 episode", type=int, default=1000)
    parser.add_argument("--random-seed", help="random seed for repeatability", type=int, default=1234)
    parser.add_argument("--render-env", help="render the env", action="store_true")
    parser.add_argument("--new-estimator", help="if creating new estimators instead of loading old ones", action="store_true")
    parser.add_argument("--max-estimators-to-keep", help="maximal number of estimators to keep checkpointing",
            type=int, default=2)
    parser.add_argument("--estimator-save-freq", help="estimator save frequency (per number of rollout steps)",
            type=int, default=10000)
    parser.add_argument("--estimator-load-mode", help="0: load most recent 1: load best", type=int, default=0)
    parser.add_argument("--replay-buffer-load-dir", help="directory for loading replay buffer")
    parser.add_argument("--replay-buffer-save-dir", help="directory for storing replay buffer")
    parser.add_argument("--replay-buffer-save-freq", help="replay buffer save frequency (per number of episodes)", type=int,
            default=500)
    parser.add_argument("--log-stats-freq", help="Stats log(tensorboard info) frequency (per number of\
                        rollout steps/transitions).\ Zero to turn off stats log", type=int, default=100)
    parser.add_argument("--eval-replay-buffer-load-dir", help="directory for loading evaluation replay buffer")

    # Agent parameters
    parser.add_argument("--num-updates", help="Number of estimator updates per training step", type=int, default=1)
    parser.add_argument("--train-freq", help="Training frequency (per number of rollout steps)", type=int, default=1)
    parser.add_argument("--num-test-eps", help="Number of test episodes", type=int, default=20)
    parser.add_argument("--test-freq", help="Testing frequency (per number of rollout steps)", type=int, default=5000)
    parser.add_argument("--normalize-states", help="If normalize states", action="store_true")
    parser.add_argument("--normalize-returns", help="If normalize returns", action="store_true")

    parser.set_defaults(stop_agent_learning=False)
    parser.set_defaults(render_env=False)
    parser.set_defaults(new_estimator=False)
    parser.set_defaults(normalize_states=True)
    parser.set_defaults(normalize_returns=False)
    return parser

if __name__ == "__main__":
    #TODO
    pass
