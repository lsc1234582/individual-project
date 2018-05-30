import argparse
from Environments.VREPEnvironments import VREPPushTask7DoFIKEnvironment
from Environments.VREPEnvironments import VREPPushTaskEnvironment
from Environments.VREPEnvironments import VREPPushTask7DoFEnvironment
from Environments.VREPEnvironments import VREPPushTask7DoFSparseRewardsIKEnvironment
import numpy as np
import tkinter as tk
from Utils import ReplayBuffer
import pprint

class Application(tk.Frame):
    w=       25
    s=       39
    a=       38
    d=       40
    q=       24
    e=       26
    num_8=   80
    num_2=   88
    num_4=   83
    num_6=   85
    num_7=   79
    num_9=   81
    up=      111
    down=    116
    left=    113
    right=   114
    f=       41
    j=       44
    c=       54
    o=       32
    r=       27
    num_5=   84
    ctrl=    105
    shift=   62
    enter=   36

    mod_key_code = [ctrl, shift]
    key_code = [w, s, a, d, q, e, num_8, num_2, num_4, num_6, num_7, num_9, up, down, left, right, f, j, c, o, r,
            num_5, enter]

    x_ind = 0
    y_ind = 1
    z_ind = 2
    rx_ind = 3
    ry_ind = 4
    rz_ind = 5
    r_w_f_ind = 6
    r_f_ind = 7
    gripper_c_ind = 8
    step_ind = 9
    reset_tt_ind = 10

    d_pos = 0.01
    d_rot = 0.1

    def __init__(self, env, replay_buffer, master=None, args=None):
        assert(env is not None)
        super().__init__(master)
        #master.bind("<Shift-Up-Release>", Application._shiftUPHandler)
        self.pack()
        self._createWidgets()
        self.action = [0 for _ in range(9)]
        master.bind("<KeyPress>", self.keydown)
        master.bind("<KeyRelease>", self.keyup)
        #master.bind("<Up>", self._upHandler)
        #master.bind("<Down>", self._downHandler)
        self._mod_keys = []
        self._keys = []
        self._rot_world_frame_fixed = -1
        self._gripper_closing = 1
        self._env = env
        self._rb = replay_buffer
        self._args = args
        self._last_state = self._env.reset().reshape((1, -1))
        self._isRecording = False

    def _keyHandler(event):
        print("pressed {}".format(repr(event.keycode)))

    def _upHandler(self, event):
        print("shift up pressed {}".format(repr(event.char)))
        self.action[0] = 1
        print(self.action)

    def keyup(self, e):
        if e.keycode in self.mod_key_code:
            history = self._mod_keys
        elif e.keycode in self.key_code:
            history = self._keys
        else:
            return
        if e.keycode in history :
            history.pop(history.index(e.keycode))

    def keydown(self, e):
        if e.keycode in self.mod_key_code:
            history = self._mod_keys
        elif e.keycode in self.key_code:
            history = self._keys
        else:
            return
        if not e.keycode in history :
            history.append(e.keycode)

        if len(self._keys) > 0:
            action = self._generateAction(self._keys, self._mod_keys)
            #print(action)
            assert(self._env is not None)
            state, rewards, done, _ = self._env.step(action)
            self._rbSizeVar.set("RB Size: {}".format(self._rb.size()))
            if state is not None:
                # NOTE: the action is idealised
                print(env.getStateString(self._last_state))
                print(env.getStateString(state))
                print(rewards)
                print(done)
                joints_dvel = state[:, :6] - self._last_state[:, :6]
                action_to_save = np.concatenate([joints_dvel,
                    np.reshape(np.array(self._gripper_closing), (1, -1))], axis=1)
                print(action_to_save)
                if self._isRecording:
                    self._rb.add(self._last_state.squeeze().copy(), action_to_save.squeeze().copy(),
                            rewards.squeeze().copy(), state.squeeze().copy(), done)
                self._last_state = state
            if done:
                self._last_state = self._env.reset().reshape((1, -1))
            #print(state)

    def _generateAction(self, keys, mod_keys):
        action = np.zeros(11)
        if self.enter in keys:
            # Step
            action[self.step_ind] = 1
        else:
            action[self.step_ind] = -1
        if self.r in keys:
            # Reset tip target
            action[self.reset_tt_ind] = 1
        else:
            action[self.reset_tt_ind] = -1
        if self.num_8 in keys:
            action[self.x_ind] += self.d_pos
        if self.num_2 in keys:
            action[self.x_ind] -= self.d_pos
        if self.num_4 in keys:
            action[self.y_ind] += self.d_pos
        if self.num_6 in keys:
            action[self.y_ind] -= self.d_pos
        if self.num_7 in keys:
            action[self.z_ind] += self.d_pos
        if self.num_9 in keys:
            action[self.z_ind] -= self.d_pos

        if self.w in keys:
            action[self.rx_ind] += self.d_rot
        if self.s in keys:
            action[self.rx_ind] -= self.d_rot
        if self.a in keys:
            action[self.ry_ind] += self.d_rot
        if self.d in keys:
            action[self.ry_ind] -= self.d_rot
        if self.q in keys:
            action[self.rz_ind] += self.d_rot
        if self.e in keys:
            action[self.rz_ind] -= self.d_rot

        if self.shift in mod_keys:
            action[self.r_w_f_ind] = 1
        else:
            action[self.r_w_f_ind] = -1

        if self.f in keys:
            self._rot_world_frame_fixed = 1
        elif self.j in keys:
            self._rot_world_frame_fixed = -1
        action[self.r_f_ind] = self._rot_world_frame_fixed

        if self.c in keys:
            self._gripper_closing = 1
        elif self.o in keys:
            self._gripper_closing = -1
        action[self.gripper_c_ind] = self._gripper_closing

        return action.reshape(1, -1)

    def _createWidgets(self):
        self._isRecordingVar = tk.StringVar()
        self._isRecordingLbl = tk.Label(self, textvariable=self._isRecordingVar).pack(side="top")

        self._rbSizeVar = tk.StringVar()
        self._rbSizeLbl = tk.Label(self, textvariable=self._rbSizeVar).pack(side="top")

        self._resetEnvBtn = tk.Button(self)
        self._resetEnvBtn["text"] = "Reset Env"
        self._resetEnvBtn["command"] = self._resetEnv
        self._resetEnvBtn.pack(side="top")

        self._startRecordingBtn = tk.Button(self)
        self._startRecordingBtn["text"] = "Start recording"
        self._startRecordingBtn["command"] = self._startRecording
        self._startRecordingBtn.pack(side="top")

        self._stopRecordingBtn = tk.Button(self)
        self._stopRecordingBtn["text"] = "Stop recording"
        self._stopRecordingBtn["command"] = self._stopRecording
        self._stopRecordingBtn.pack(side="top")

        self._clearRecordingBtn = tk.Button(self)
        self._clearRecordingBtn["text"] = "Clear recording"
        self._clearRecordingBtn["command"] = self._clearRecording
        self._clearRecordingBtn.pack(side="top")

        self._saveRecordingBtn = tk.Button(self)
        self._saveRecordingBtn["text"] = "Save recording"
        self._saveRecordingBtn["command"] = self._saveRecording
        self._saveRecordingBtn.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")

    def _startRecording(self):
        self._isRecording = True
        self._isRecordingVar.set("Recording started...")

    def _stopRecording(self):
        self._isRecording = False
        self._isRecordingVar.set("Recording stopped...")

    def _clearRecording(self):
        self._rb.clear()
        self._rbSizeVar.set("RB Size: {}".format(self._rb.size()))

    def _saveRecording(self):
        print("Saving replay buffer")
        self._rb.save(self._args.replay_buffer_save_dir)
        print("Replay buffer saved")

    def _loadRecording(self):
        pass

    def _resetEnv(self):
        print("Resetting environment")
        self._env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay buffer player")

    #parser.add_argument("--env-name", help="choose the env[VREPPushTask, Pendulum-v0]", required=True)
    parser.add_argument("--replay-buffer-save-dir", help="directory for saving replay buffer", required=True)
    parser.add_argument("--replay-buffer-load-dir", help="directory for loading replay buffer")
    parser.add_argument("--replay-buffer-size-log", help="max size of the replay buffer as exponent of 10", type=int, default=6)
    args = parser.parse_args()

    replay_buffer = ReplayBuffer(10 ** args.replay_buffer_size_log)
    if args.replay_buffer_load_dir is not None:
        replay_buffer.load(args.replay_buffer_load_dir)

    with VREPPushTask7DoFSparseRewardsIKEnvironment(mico_model_path="models/robots/non-mobile/MicoRobot7DoFIK.ttm") as env:
        #env.reset()
        root = tk.Tk()
        app = Application(master=root, env=env, replay_buffer=replay_buffer, args=args)
        app.mainloop()
    #with VREPPushTaskIKEnvironment(mico_model_path="models/robots/non-mobile/MicoRobotIK2.ttm") as env:
    #    env.reset()
    #    _ = input()
    #with VREPPushTaskEnvironment() as env:
    #    state = env.reset()
    #    while True:
    #        print("Current state:")
    #        print(state)
    #        action_str = input("Action (a sequence of floats, separated by space|reset|quit|[Enter for same entry as
    #                previous one]):")
    #        if action_str == "quit":
    #            break
    #        elif action_str == "reset":
    #            state = env.reset()
    #        else:
    #            action = np.array(list(map(float, action_str.split())))
    #            state = env.step(action.reshape(1, -1))

