import argparse
import numpy as np
from Utils.Utils import ReplayBuffer
from Utils.Utils import generateRandomAction
from EnvironmentFactory import EnvironmentContext

def runEnvironmentWithAgent(args):
    rb = ReplayBuffer(10 ** args.replay_buffer_size_log)
    if args.replay_buffer_load_dir is not None:
        rb.load(args.replay_buffer_load_dir)
    with EnvironmentContext(args.env_name, port=19997) as env:
        # To record progress across different training sessions
        # Run the environment feedback loop
        assert(np.all(np.array(env.action_space.high) == - np.array(env.action_space.low)))
        for episode_num in range(args.num_episodes):
            state = env.reset()
            for step in range(args.max_episode_length):
                # NOTE: Assume uniform high
                action = generateRandomAction(env.action_space.high[0], env.action_space.shape[0])
                next_state, reward, done, _ = env.step(action.reshape(1, -1))
                if step == args.max_episode_length - 1:
                    done = True
                rb.add(state.squeeze(), action.squeeze(), reward.squeeze(), next_state.squeeze(), done)
                state = np.copy(next_state)
                #logger.debug("Observation")
                #logger.debug(observation)
                #logger.debug("Action")
                #logger.debug(action)
                #logger.debug("Reward")
                #logger.debug(reward)
                #logger.debug("Done")
                #logger.debug(done)
            # No need to push forward when the agent stops training and has collected enough episodes to obtain a score

        rb.save(args.replay_buffer_save_dir)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Generate random replay buffer")

    parser.add_argument("--env-name", help="choose the env[VREPPushTask, Pendulum-v0]", required=True)
    parser.add_argument("--replay-buffer-load-dir", help="directory for loading replay buffer")
    parser.add_argument("--replay-buffer-save-dir", help="directory for saving replay buffer", required=True)
    parser.add_argument("--replay-buffer-size-log", help="replay buffer size, as an exponent of 10", type=int, required=True)
    parser.add_argument("--max-episode-length", help="max episode length", type=int,
            default=100)
    parser.add_argument("--num-episodes", help="Number of episodes to play", type=int,
            default=10)
    args = parser.parse_args()
    runEnvironmentWithAgent(args)

