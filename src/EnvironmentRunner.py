import tensorflow as tf
from EnvironmentFactory import EnvironmentContext
from Utils import getModuleLogger

# Module logger
logger = getModuleLogger(__name__)


def runEnvironmentWithAgent(makeAgent, args):
    logger.info("Run info:")
    logger.info(vars(args))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    logger.info("Making environment {}".format(args.env_name))
    with EnvironmentContext(args.env_name) as env, tf.Session(config=config) as session:
        #global_step = tf.Variable(0, name="global_step", trainable=False)
        # To record progress across different training sessions
        global_episode_num = tf.Variable(1, name="global_episode_num", trainable=False)
        logger.info("Making agent {}".format(args.agent_name))
        agent = makeAgent(session, env, args)

        session.run(tf.global_variables_initializer())
        if args.new_estimator:
            logger.info("Saving initial agent to {}".format(args.estimator_dir))
            agent.save(args.estimator_dir, 0, True)
        else:
            logger.info("Restoring agent from {}".format(args.estimator_dir))
            agent.load(args.estimator_dir)

        episode_start = session.run(global_episode_num)
        logger.info("Continueing at episode {}".format(episode_start))
        # Run the environment feedback loop
        for episode_num in range(episode_start, args.num_episodes + 1):
            observation = env.reset()
            reward = 0.0
            done = False
            action, done = agent.act(observation, reward, done, episode_num, is_learning=(not args.stop_agent_learning))

            while not done:
                if args.render_env:
                    env.render()
                observation, reward, done, _ = env.step(action)
                action, done = agent.act(observation, reward, done, episode_num, is_learning=(not args.stop_agent_learning))
                #logger.debug("Observation")
                #logger.debug(observation)
                #logger.debug("Action")
                #logger.debug(action)
                #logger.debug("Reward")
                #logger.debug(reward)
                #logger.debug("Done")
                #logger.debug(done)
            # Increment global episode num
            session.run(tf.assign(global_episode_num, global_episode_num + 1))
            if not args.stop_agent_learning and episode_num % args.estimator_save_freq == 0:
                logger.info("Saving agent checkpoints")
                agent.save(args.estimator_dir, episode_num, False)
            # No need to push forward when the agent stops training and has collected enough episodes to obtain a score
            if agent._stop_training and agent.score():
                logger.warn("Agent stopped training. Exiting experiment...")
                break
        if not args.stop_agent_learning:
            logger.info("Saving last agent to {}".format(args.estimator_dir))
            agent.save(args.estimator_dir, episode_num, False)

    logger.info("Best score: {}".format(agent.score()))
    logger.info("Exiting environment: {}".format(args.env_name))
    return agent.score()

if __name__ == "__main__":
    #TODO
    pass
