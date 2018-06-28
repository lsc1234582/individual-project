from Environments.VREPEnvironments import VREPPushTaskEnvironment
import numpy as np



if __name__ == "__main__":
    with VREPPushTaskEnvironment(
                init_cb_pos=[0.3, 0.5, 0.05],
                init_tg_pos=[0.3, 0.8, 0.002],
            ) as env:
        state = env.reset()
        while True:
            print("Current state:")
            print(state)
            init_pos_str = input("Initial pos (a sequence of floats, separated by space|quit):")
            if init_pos_str == "quit":
                break
            else:
                init_pos = np.array(list(map(float, init_pos_str.split())))
                state = env.reset()

