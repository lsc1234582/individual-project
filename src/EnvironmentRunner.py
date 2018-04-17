import collections
import tensorflow as tf
from EnvironmentFactory import EnvironmentContext
from Utils import getModuleLogger
from Utils import SortedDisplayDict

# Module logger
logger = getModuleLogger(__name__)


def runEnvironmentWithAgent(makeAgent, args):
    logger.info("Run info:")
    logger.info(SortedDisplayDict(vars(args)))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    logger.info("Making environment {}".format(args.env_name))
    # Set graph-level random seed to ensure repeatability of experiments
    tf.set_random_seed(args.random_seed)
    with EnvironmentContext(args.env_name) as env, tf.Session(config=config) as session:
        #global_step = tf.Variable(0, name="global_step", trainable=False)
        # To record progress across different training sessions
        global_episode_num = tf.Variable(0, name="global_episode_num", trainable=False)
        logger.info("Making agent {}".format(args.agent_name))
        agent = makeAgent(session, env, args)

        session.run(tf.global_variables_initializer())
        if args.new_estimator:
            logger.info("Saving session meta file to {}".format(args.estimator_dir))
            agent.save(args.estimator_dir, step=0, write_meta_graph=True)
        else:
            logger.info("Restoring agent from {}".format(args.estimator_dir))
            agent.load(args.estimator_dir, is_best=(args.estimator_load_mode==1))

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

if __name__ == "__main__":
    #TODO
    pass
