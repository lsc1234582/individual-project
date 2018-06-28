"""
A rather crude replay buffer player
"""
import os
import argparse
import numpy as np
import pandas as pd
from Utils import ReplayBuffer
from EnvironmentFactory import EnvironmentContext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay buffer player")

    parser.add_argument("--env-name", help="choose the env[VREPPushTask, Pendulum-v0]", required=True)
    parser.add_argument("--replay-buffer-load-dir", help="directory for loading replay buffer")
    parser.add_argument("--replay-buffer-save-dir", help="directory for saving replay buffer")
    parser.add_argument("--replay-buffer-size-log", help="Size", type=int, default=6)
    parser.add_argument("--max-episode-length", help="max episode length", type=int,
            default=100)
    parser.add_argument("--num-episodes", help="Number of episodes to play", type=int,
            default=10)
    parser.add_argument("--joint-vel-csv-dir", help="Path to csv file holding joint velocities data")
    parser.add_argument("--action-npy-dir", help="Path to npy file holding action data")
    args = parser.parse_args()
    episode_return = 0
    episode_steps = 0
    get_rb_stats = False
    episode_num = 1
    if args.joint_vel_csv_dir is not None:
        #assert args.replay_buffer_save_dir is not None
        if args.replay_buffer_save_dir is not None:
            assert not os.path.exists(args.replay_buffer_save_dir)
        rb_to_save = ReplayBuffer(10 ** args.replay_buffer_size_log)
        if args.replay_buffer_load_dir is not None:
            rb_to_save.load(args.replay_buffer_load_dir)

        joint_vels = np.array(pd.read_csv(args.joint_vel_csv_dir, header=None))

        # Interpolate velocities
        new_joint_vels = np.copy(joint_vels[:-1, 1:])
        t0 = 0
        st = joint_vels[0, 0]
        t1 = joint_vels[1, 0]
        v0 = 0
        v1 = joint_vels[1, 1:7]

        a = (st - t0)/(t1-t0)
        a_ = 1 - a
        new_joint_vels[0, :6] = v0 * a_ + v1 * a
        for i in range(1, new_joint_vels.shape[0]):
            t0 = joint_vels[i - 1, 0]
            st = joint_vels[i, 0]
            t1 = joint_vels[i + 1, 0]
            v0 = joint_vels[i - 1, 1:7]
            v1 = joint_vels[i + 1, 1:7]
            a = (st - t0)/(t1-t0)
            a_ = 1 - a
            new_joint_vels[i, :6] = v0 * a_ + v1 * a
        #print(new_joint_vels)

        joint_vels = joint_vels[:, 1:]
        corrected_actions = []
        with EnvironmentContext(args.env_name) as env:
            done = False
            #print("HERE")
            state = env.reset()
            #print("HERE2")
            for step in range(new_joint_vels.shape[0] - 1):
                #print("Actual next vel")
                #print(joint_vels[step+1, :6])
                action = np.zeros((1, joint_vels.shape[1]))
                ##print(action.shape)
                action[0, :6] = joint_vels[step + 1, :6] - joint_vels[step, :6]
                action[0, 6] = joint_vels[step, 6]
                #print("Action")
                #print(action)
                next_state, r, _, done, _, corrected_action = env.step(new_joint_vels[step+1, :].reshape(1, -1), vels_only=True)
                rb_to_save.add(state.squeeze(), corrected_action.squeeze(), r.squeeze(), next_state.squeeze(), done)
                episode_return += r
                episode_steps += 1
                state = np.copy(next_state)
                #corrected_actions.append(corrected_action)
                #_, r, done, _ = env.step(action.reshape(1, -1), vels_only=False)
                #print("Reward: {}".format(r))
                if done:
                    env.reset()
                    print("Episode return: {}".format(episode_return))
                    episode_return = 0
                    episode_steps = 0
                    episode_num += 1
                    break

            #np.save("/home/sicong/vrep_actions.npy", np.concatenate(corrected_actions, axis=0))
            if args.replay_buffer_save_dir:
                rb_to_save.save(args.replay_buffer_save_dir)
    elif args.replay_buffer_load_dir is not None:
        replay_buffer = ReplayBuffer(1)
        replay_buffer.load(args.replay_buffer_load_dir)

        if args.replay_buffer_save_dir is not None:
            assert not os.path.exists(args.replay_buffer_save_dir)
        rb_to_save = ReplayBuffer(replay_buffer._maxsize)

        print("replay buffer size")
        print(replay_buffer.size())
        print("number of episodes in the replay buffer")
        print(int(replay_buffer.size() / args.max_episode_length))
        if get_rb_stats:
            for step, exp in enumerate(replay_buffer._storage):
                _, action, r, _, done = exp
                episode_return += r
                episode_steps += 1
                #print("Reward: {}".format(r))
                if done:
                    print("Episode {}: return: {}; steps: {}".format(episode_num, episode_return, episode_steps))
                    episode_return = 0
                    episode_steps = 0
                    episode_num += 1
        else:
            with EnvironmentContext(args.env_name) as env:
                done = False
                state = env.reset()
                for step, exp in enumerate(replay_buffer._storage):
                    _, action, _, _, done = exp
                    next_state, r, _,  done2, _ = env.step(np.reshape(action, (1, -1)))
                    episode_return += r
                    episode_steps += 1
                    rb_to_save.add(state.squeeze(), action.squeeze(), r.squeeze(), next_state.squeeze(), done2)
                    #print("Reward: {}".format(r))
                    #_ = input()
                    state = np.copy(next_state)
                    if done:
                        env.reset()
                        print("Episode {}: return: {}; steps: {}".format(episode_num, episode_return, episode_steps))
                        episode_return = 0
                        episode_steps = 0
                        episode_num += 1

                if args.replay_buffer_save_dir:
                    rb_to_save.save(args.replay_buffer_save_dir)

    elif args.action_npy_dir is not None:
        actions = np.load(args.action_npy_dir)
        with EnvironmentContext(args.env_name) as env:
            done = False
            #print("HERE")
            env.reset()
            #print("HERE2")
            for step in range(actions.shape[0]):
                #print("Actual next vel")
                #print(joint_vels[step+1, :6])
                #print(action)
                _, r, done, _ = env.step(actions[step, :].reshape(1, -1))
                #_, r, done, _ = env.step(action.reshape(1, -1), vels_only=False)
                print("Reward: {}".format(r))
                if done:
                    env.reset()
                    episode_num += 1

    else:
        print("Must provide either a csv file containing the joint velocities or the replay buffer.")

