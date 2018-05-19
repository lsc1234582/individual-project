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
    with EnvironmentContext(args.env_name) as env, tf.Session(config=config) as session:
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
                if args.render_env:
                    env.render()
                observation, reward, done, _ = env.step(action)
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
    parser = argparse.ArgumentParser(description="provide arguments for DPGAC2PEH1VEH1 agent")

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
    parser.add_argument("--log-stats-freq", help="Stats log frequency (per number of episodes", type=int,
            default=1)

    parser.set_defaults(stop_agent_learning=False)
    parser.set_defaults(render_env=False)
    parser.set_defaults(new_estimator=False)
    return parser

if __name__ == "__main__":
    #TODO
    pass
