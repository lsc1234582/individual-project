"""
Create environment.

New environments need to be registered here.
"""
import gym
import numpy as np
import Environments.VREPEnvironments as VREPEnvironments

ENVS = {
        "Pendulum-v0": gym.make,
        "VREPPushTask": VREPEnvironments.make,
        "VREPPushTask2": VREPEnvironments.make,
        "VREPPushTask3": VREPEnvironments.make,
        "VREPPushTask4": VREPEnvironments.make,
        "VREPPushTaskContact": VREPEnvironments.make,
        "VREPPushTaskContact2": VREPEnvironments.make,
        "VREPPushTask7DoF": VREPEnvironments.make,
        "VREPPushTask7DoF2": VREPEnvironments.make,
        "VREPPushTask7DoF3": VREPEnvironments.make,
        "VREPPushTask7DoF4": VREPEnvironments.make,
        "VREPPushTask7DoFIK": VREPEnvironments.make,
        "VREPPushTask7DoFSparseRewards": VREPEnvironments.make,
        "VREPPushTask7DoFSparseRewardsIKEnvironment": VREPEnvironments.make,
        "VREPGraspTask7DoF": VREPEnvironments.make,
        "VREPGraspTask7DoFSparseRewards": VREPEnvironments.make,
        "VREPGraspTask7DoFSparseRewardsIK": VREPEnvironments.make,
        }

def MakeEnvironment(env_name, *args, **kwargs):
    return (ENVS[env_name])(env_name, *args, **kwargs)

class EnvironmentContext(object):
    def __init__(self, env_name, *args, **kwargs):
        self._env_name = env_name
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        self._env = MakeEnvironment(self._env_name, *self._args, **self._kwargs)
        return self._env

    def __exit__(self, exception_type, exception_value, traceback):
        self._env.close()
        return False

if __name__ == "__main__":
    # Simple interactive program to test/debug environment
    with EnvironmentContext("VREPPushTask7DoF") as env:
        print("resetting")
        env.reset()
        print("resetted")
        for _ in range(50):
            state, reward, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(1, -1))
            print("state")
            print(state)
            print("reward")
            print(reward)
            _ = input()
