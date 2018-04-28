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
        "VREPPushTaskMultiStepReward": VREPEnvironments.make,
        "VREPPushTaskMultiStepRewardContact2": VREPEnvironments.make,
        }

def MakeEnvironment(env_name):
    return (ENVS[env_name])(env_name)

class EnvironmentContext(object):
    def __init__(self, env_name):
        self._env_name = env_name

    def __enter__(self):
        self._env = MakeEnvironment(self._env_name)
        return self._env

    def __exit__(self, exception_type, exception_value, traceback):
        self._env.close()
        return False

if __name__ == "__main__":
    # Simple interactive program to test/debug environment
    with EnvironmentContext("VREPPushTaskMultiStepReward") as env:
        env.reset()
        for _ in range(50):
            state, reward, _, _ = env.step(np.zeros((1, 6)))
            print("state")
            print(state)
            print("reward")
            print(reward)
            _ = input()
