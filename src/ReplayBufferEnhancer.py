import argparse
import numpy as np
import sys
import pprint
sys.path.insert(0, "/home/sicong/Projects/FourthYearProjects/IndividualProject/individual-project/src")
from Utils import ReplayBuffer

def injectGaussianNoise(resample_amount, original_rb, enriched_rb, mean, std):
    # Inject zero-mean gaussian noise into the current state, action, reward and next state data
    # Note that the variance is relative to the variance of each data point/dimension within the original
    # replay buffer and is fixed at std at standard normal dist.
    state_std, action_std, reward_std = original_rb.std()
    #print(original_rb._buffer)

    for _ in range(resample_amount):
        for i, exp in enumerate(original_rb._buffer):
            #print("{}".format(i))
            #pprint.pprint(exp)
            current_state, action, reward, termination, next_state = exp
            current_state = np.copy(current_state)
            action = np.copy(action)
            reward = np.copy(reward)
            termination = np.copy(termination)
            next_state = np.copy(next_state)
            #print("haha")
            #print(reward)
            #print("haha2")
            #print(reward_std)
            current_state += np.random.normal(mean, std, current_state.shape) * state_std
            action += np.random.normal(mean, std, action.shape) * action_std
            reward += np.random.normal(mean, std, reward.shape) * reward_std
            next_state += np.random.normal(mean, std, next_state.shape) * state_std

            enriched_rb.add(current_state, action, reward, termination, next_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="provide arguments for DPGAC2PEH1VEH1 agent")

    parser.add_argument("--resample-amount", help="directory for loading replay buffer", type=int, required=True)
    parser.add_argument("--replay-buffer-load-dir", help="directory for loading replay buffer", required=True)
    parser.add_argument("--replay-buffer-save-dir", help="directory for storing replay buffer", required=True)
    parser.add_argument("--replay-buffer-size-log", help="max size of the replay buffer as exponent of 10", type=int,
            default=6)
    args = parser.parse_args()

    replay_buffer = ReplayBuffer(10 ** args.replay_buffer_size_log)
    replay_buffer.load(args.replay_buffer_load_dir)

    resampled_replay_buffer = ReplayBuffer(10 ** args.replay_buffer_size_log)
    resampled_replay_buffer.load(args.replay_buffer_load_dir)

    print("replay buffer size")
    print(replay_buffer.size())
    print("last entry")
    print(replay_buffer._buffer[-1])
    injectGaussianNoise(args.resample_amount, replay_buffer, resampled_replay_buffer, 0, 1e-2)

    resampled_replay_buffer.save(args.replay_buffer_save_dir)

