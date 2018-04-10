import tensorflow as tf
from EnvironmentFactory import EnvironmentContext

def runEnvironmentWithAgent(makeAgent, args):
    tf.logging.info("Run info:")
    tf.logging.info(vars(args))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    tf.logging.info("Making environment {}".format(args.env_name))
    with EnvironmentContext(args.env_name) as env, tf.Session(config=config) as session:
        global_step = tf.Variable(0, name="global_step", trainable=False)
        tf.logging.info("Making agent {}".format(args.agent_name))
        agent = makeAgent(session, env, args)
        session.run(tf.global_variables_initializer())

        # Run the environment feedback loop
        for episode_num in range(1, args.num_episodes + 1):
            observation = env.reset()
            reward = 0.0
            done = False
            action, done = agent.act(observation, reward, done, episode_num, is_learning=args.is_agent_learning)

            while not done:
                if args.render_env:
                    env.render()
                observation, reward, done, _ = env.step(action)
                action, done = agent.act(observation, reward, done, episode_num, is_learning=args.is_agent_learning)
                tf.logging.debug("Observation")
                tf.logging.debug(observation)
                tf.logging.debug("Action")
                tf.logging.debug(action)
                tf.logging.debug("Reward")
                tf.logging.debug(reward)
                tf.logging.debug("Done")
                tf.logging.debug(done)
            # No need to push forward when the agent stops training and has collected enough episodes to obtain a score
            if agent._stop_training and agent.score():
                tf.logging.warn("Agent stopped training. Exiting experiment...")
                break

    tf.logging.info("Best score: {}".format(agent.score()))
    tf.logging.info("Exiting environment: {}".format(args.env_name))
    return agent.score()

if __name__ == "__main__":
    #TODO
    pass
