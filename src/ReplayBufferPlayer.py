"""
A rather crude replay buffer player
"""
import argparse
import numpy as np
from Utils import ReplayBuffer
from EnvironmentFactory import EnvironmentContext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay buffer player")

    parser.add_argument("--env-name", help="choose the env[VREPPushTask, Pendulum-v0]", required=True)
    parser.add_argument("--replay-buffer-load-dir", help="directory for loading replay buffer", required=True)
    parser.add_argument("--max-episode-length", help="max episode length", type=int,
            default=100)
    parser.add_argument("--num-episodes", help="Number of episodes to play", type=int,
            default=10)
    args = parser.parse_args()

    replay_buffer = ReplayBuffer(1)
    replay_buffer.load(args.replay_buffer_load_dir)

    print("replay buffer size")
    print(replay_buffer.size())
    print("number of episodes in the replay buffer")
    print(int(replay_buffer.size() / args.max_episode_length))

    episode_num = 1
    with EnvironmentContext(args.env_name) as env:
        done = False
        env.reset()
        for step, exp in enumerate(replay_buffer.iter()):
            _, action, _, done, _ = exp
            env.step(np.reshape(action, (1, -1)))
            if done:
                env.reset()
                episode_num += 1
