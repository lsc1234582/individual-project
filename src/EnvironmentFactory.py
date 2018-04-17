"""
Create environment.

New environments need to be registered here.
"""
import gym
import Environments.VREPEnvironments as VREPEnvironments

ENVS = {
        "Pendulum-v0": gym.make,
        "VREPPushTask": VREPEnvironments.make,
        "VREPPushTaskContact": VREPEnvironments.make,
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

