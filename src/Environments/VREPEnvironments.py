import numpy as np
import tensorflow as tf
import vrep

from Utils import getModuleLogger

# Module logger
logger = getModuleLogger(__name__)

class Box(object):
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high

class VREPPushTaskEnvironment(object):
    """
    Distance unit: m
    Maximum distance between target and cuboid: 2m; this will affect the reward function for VREPPushTaskMultiStepRewardEnvironment
    """
    MAX_JOINT_VELOCITY_DELTA = 1.0
    MAX_JOINT_VELOCITY = 6.0
    CUBOID_SIDE_LENGTH = 0.1
    GRIPPER_BASE_TO_CLOSED_TIP_DIST = 0.15
    DEFAULT_JOINT_POSITIONS = [np.pi, 1.5 * np.pi, 1.5 * np.pi, np.pi, np.pi, np.pi]
    DEFAULT_CUBOID_POSITION = [0.3, 0.5, 0.05]
    DEFAULT_CUBOID_ORIENTATION = [0., 0., 0.]
    DEFAULT_TARGET_POSITION = [0.3, 0.8, 0.002]
    # Simulation delta time in seconds
    SIMULATION_DT = 0.05

    def __init__(self, port=19991, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None,
                mico_model_path="models/robots/non-mobile/MicoRobot.ttm"):
        logger.info("Creating VREPPushTaskEnvironment")
        self._init_joint_pos = init_joint_pos if not init_joint_pos is None else VREPPushTaskEnvironment.DEFAULT_JOINT_POSITIONS
        self._init_cb_pos = init_cb_pos if not init_cb_pos is None else VREPPushTaskEnvironment.DEFAULT_CUBOID_POSITION
        self._init_cb_orient = init_cb_orient if not init_cb_orient is None else\
                                VREPPushTaskEnvironment.DEFAULT_CUBOID_ORIENTATION
        self._init_tg_pos = init_tg_pos if not init_tg_pos is None else VREPPushTaskEnvironment.DEFAULT_TARGET_POSITION
        self.action_space = Box((6,), (-1.0,), (1.0,))
        self.observation_space = Box((24,), (-999.0,), (999.0,))

        vrep.simxFinish(-1) # just in case, close all opened connections
        self.client_ID=vrep.simxStart('127.0.0.1',port,True,True,5000,5) # Connect to V-REP
        if self.client_ID == -1:
            raise IOError("VREP connection failed.")

         # enable the synchronous mode on the client:
        vrep.simxSynchronous(self.client_ID,True)
        # start the simulation:
        vrep.simxStartSimulation(self.client_ID, vrep.simx_opmode_blocking)
        self._reset_yet = False
        self.mico_model_path = mico_model_path

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        return False


    def _tearDownDatastream(self):
        # tear down datastreams
        for i in range(6):
            _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.joint_handles[i], 2012, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetJointPosition(self.client_ID, self.joint_handles[i],
                    vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.cuboid_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.target_plane_handle, -1, vrep.simx_opmode_discontinue)

    def close(self):
        logger.info("Closing VREPPushTaskEnvironment")

        #self._tearDownDatastream()

        # stop the simulation:
        vrep.simxStopSimulation(self.client_ID, vrep.simx_opmode_blocking)
        # before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(self.client_ID)
        # disconnect
        vrep.simxFinish(self.client_ID)

    def getCurrentState(self, client_ID, joint_handles, gripper_handle, cuboid_handle, target_plane_handle):
        """
            TODO: Refactor away arguments
            Return the state as an array of shape (24, )
            [current_vel, joint_angles, gripper_pos, gripper_orient, cuboid_gripper_vec, target_plane_cuboid_vec]
             6              6               3           3               3                   3
        """
        current_vel = np.array([0, 0, 0, 0, 0, 0], dtype='float')
        joint_angles = np.array([0, 0, 0, 0, 0, 0], dtype='float')
        # obtain first state
        for i in range(6):
            ret, current_vel[i] = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012,
                    vrep.simx_opmode_buffer)
            while ret != vrep.simx_return_ok:
                ret, current_vel[i] = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012,
                        vrep.simx_opmode_buffer)
            ret, joint_angles[i] = vrep.simxGetJointPosition(client_ID, joint_handles[i], vrep.simx_opmode_buffer)
            while ret != vrep.simx_return_ok:
                ret, joint_angles[i] = vrep.simxGetJointPosition(client_ID, joint_handles[i], vrep.simx_opmode_buffer)
        ret, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        ret, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        gripper_pos = np.array(gripper_pos)
        gripper_orient = np.array(gripper_orient)

        ret, cuboid_pos = vrep.simxGetObjectPosition(client_ID, cuboid_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, cuboid_pos = vrep.simxGetObjectPosition(client_ID, cuboid_handle, -1, vrep.simx_opmode_buffer)
        cuboid_pos = np.array(cuboid_pos)

        ret, target_plane_pos = vrep.simxGetObjectPosition(client_ID, target_plane_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, target_plane_pos = vrep.simxGetObjectPosition(client_ID, target_plane_handle, -1, vrep.simx_opmode_buffer)
        target_plane_pos = np.array(target_plane_pos)

        cuboid_gripper_vec = cuboid_pos - gripper_pos
        target_plane_cuboid_vec = target_plane_pos - cuboid_pos

        return np.concatenate([current_vel, joint_angles, gripper_pos, gripper_orient, cuboid_gripper_vec,
            target_plane_cuboid_vec])

    def _loadModelsAndAssignHandles(self):
        # get handles
        _, self.cuboid_handle = vrep.simxGetObjectHandle(self.client_ID, 'Cuboid', vrep.simx_opmode_blocking)
        _, self.target_plane_handle = vrep.simxGetObjectHandle(self.client_ID, 'TargetPlane', vrep.simx_opmode_blocking)

        _, self.model_base_handle = vrep.simxLoadModel(self.client_ID, self.mico_model_path, 0, vrep.simx_opmode_blocking)
        self.joint_handles = [-1, -1, -1, -1, -1, -1]
        for i in range(6):
            _, self.joint_handles[i] = vrep.simxGetObjectHandle(self.client_ID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
        _, self.gripper_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand', vrep.simx_opmode_blocking)

    def _initialiseScene(self):
        """
        Initialise the environment
        """
        # initialise mico joint positions, cuboid orientation and cuboid position
        vrep.simxPauseCommunication(self.client_ID, 1)
        for i in range(6):
            ret = vrep.simxSetJointPosition(self.client_ID, self.joint_handles[i], self._init_joint_pos[i], vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectOrientation(self.client_ID, self.cuboid_handle, -1, self._init_cb_orient, vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectPosition(self.client_ID, self.cuboid_handle, -1, self._init_cb_pos, vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectPosition(self.client_ID, self.target_plane_handle, -1, self._init_tg_pos, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.client_ID, 0)

    def _setupDatastream(self):
        current_vel = np.array([0, 0, 0, 0, 0, 0], dtype='float')
        joint_angles = np.array([0, 0, 0, 0, 0, 0], dtype='float')

        # set up datastreams
        for i in range(6):
            _, current_vel[i] = vrep.simxGetObjectFloatParameter(self.client_ID, self.joint_handles[i], 2012,
                    vrep.simx_opmode_streaming)
            _, joint_angles[i] = vrep.simxGetJointPosition(self.client_ID, self.joint_handles[i], vrep.simx_opmode_streaming)
        _, gripper_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_streaming)
        _, gripper_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_streaming)
        _, cuboid_pos = vrep.simxGetObjectPosition(self.client_ID, self.cuboid_handle, -1, vrep.simx_opmode_streaming)
        _, target_plane_pos = vrep.simxGetObjectPosition(self.client_ID, self.target_plane_handle, -1, vrep.simx_opmode_streaming)

        # destroy dummy arrays for setting up the datastream
        del current_vel, joint_angles, gripper_pos, gripper_orient, cuboid_pos, target_plane_pos

    def reset(self):
        """
        Reset the environment
        Return initial state and assign the internal initial state
        """

        if self._reset_yet:
            self._tearDownDatastream()
            # remove Mico
            vrep.simxRemoveModel(self.client_ID, self.model_base_handle, vrep.simx_opmode_blocking)
        else:
            self._reset_yet = True

        self._loadModelsAndAssignHandles()
        self._initialiseScene()
        self._setupDatastream()
        # obtain the first state before the first time step
        # this extra synchronisation is necessary for reset
        vrep.simxSynchronousTrigger(self.client_ID)
        vrep.simxGetPingTime(self.client_ID)
        current_state = self.getCurrentState(self.client_ID, self.joint_handles, self.gripper_handle, self.cuboid_handle,
                self.target_plane_handle)

        self.state = current_state

        return current_state

    def getRewards(state, action):
        """
            Return the sum of the Euclidean distance between gripper and cuboid and the Euclidean distance between cuboid and targetPlane.
            NB: Rewards should be non-negative
        """
        return -(np.sqrt(np.sum(np.square(state[-6:-3]))) + np.sqrt(np.sum(np.square(state[-3:]))))
        #return -(np.sqrt(np.sum(np.square(action))))
        #return np.tanh(-(np.sqrt(np.sum(np.square(state[:1]))))/10.0) + 1.0



    def step(self, actions):
        """
        Execute sequences of actions (None, 6) in the environment
        Return sequences of subsequent states and rewards
        """
        next_states = []
        rewards = []
        for i in range(actions.shape[0]):
            current_vel = self.state[:6] + actions[i, :]
            vrep.simxPauseCommunication(self.client_ID, 1)
            for j in range(6):
                # Cap at max velocity
                vel = max(-VREPPushTaskEnvironment.MAX_JOINT_VELOCITY, min(VREPPushTaskEnvironment.MAX_JOINT_VELOCITY,
                    current_vel[j]))
                vrep.simxSetJointTargetVelocity(self.client_ID, self.joint_handles[j], vel,
                        vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.client_ID, 0)
            vrep.simxSynchronousTrigger(self.client_ID)
            # make sure all commands are exeucted
            vrep.simxGetPingTime(self.client_ID)
            # obtain next state
            next_state = self.getCurrentState(self.client_ID, self.joint_handles, self.gripper_handle, self.cuboid_handle,
                    self.target_plane_handle)
            next_states.append(next_state)
            rewards.append(self.__class__.getRewards(self.state, actions[i]))
            self.state = np.copy(next_state)

        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        return next_states, rewards, False, None

class VREPPushTaskMultiStepRewardEnvironment(VREPPushTaskEnvironment):

    def __init__(self, port=19991, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None):
        super().__init__(port, init_joint_pos, init_cb_pos, init_cb_orient, init_tg_pos)

    def getRewards(state, action):
        gripper_cube_dist = np.sqrt(np.sum(np.square(state[-6:-3])))
        if gripper_cube_dist >= 0.1 + VREPPushTaskEnvironment.GRIPPER_BASE_TO_CLOSED_TIP_DIST + (VREPPushTaskEnvironment.CUBOID_SIDE_LENGTH * np.sqrt(2) / 2):
            return -gripper_cube_dist
        return -gripper_cube_dist + 1 - np.sqrt(np.sum(np.square(state[-3:])))

class VREPPushTaskNonIKEnvironment(VREPPushTaskEnvironment):
    """
    Distance unit: m
    Maximum distance between target and cuboid: 2m; this will affect the reward function for VREPPushTaskMultiStepRewardEnvironment
    """
    # Reset time in seconds
    RESET_TIME = 1.2

    def __init__(self, port=19991, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None,
                mico_model_path="models/robots/non-mobile/MicoRobot.ttm"):
        super().__init__(port, init_joint_pos, init_cb_pos, init_cb_orient, init_tg_pos, mico_model_path)
        self.action_space = Box((7,), (-1.0,), (1.0,))
        self.observation_space = Box((28,), (-999.0,), (999.0,))
        self._gripper_closing = True
        self._gripper_closing_vel = -0.04

    def _tearDownDatastream(self):
        # tear down datastreams
        super()._tearDownDatastream()
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f1_handle, 2012,
                vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f1_handle, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f2_handle, 2012,
                vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f2_handle, vrep.simx_opmode_discontinue)

    def getCurrentState(self, client_ID, joint_handles, gripper_handle, cuboid_handle, target_plane_handle):
        """
            TODO: Refactor away arguments
            Return the state as an array of shape (28, )
            [joint_vel, joint_angles, gripper_pos, gripper_orient, cuboid_gripper_vec, target_plane_cuboid_vec,
             gripper_joint_vel,  gripper_joint_angles, ]
             6              6               3           3               3                   3
             2                      2
        """
        state = super().getCurrentState(client_ID, joint_handles, gripper_handle, cuboid_handle, target_plane_handle)
        # Gripper joints
        gripper_joint_vel = np.array([0.0, 0.0], dtype='float')
        gripper_joint_angles = np.array([0.0, 0.0], dtype='float')

        ret, gripper_joint_vel[0] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f1_handle, 2012,
                vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_vel[0] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f1_handle, 2012,
                    vrep.simx_opmode_buffer)
        ret, gripper_joint_vel[1] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f2_handle, 2012,
                vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_vel[1] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f2_handle, 2012,
                    vrep.simx_opmode_buffer)
        ret, gripper_joint_angles[0] = vrep.simxGetJointPosition(client_ID, self.gripper_f1_handle, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_angles[0] = vrep.simxGetJointPosition(client_ID, self.gripper_f1_handle, vrep.simx_opmode_buffer)
        ret, gripper_joint_angles[1] = vrep.simxGetJointPosition(client_ID, self.gripper_f2_handle, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_angles[1] = vrep.simxGetJointPosition(client_ID, self.gripper_f2_handle, vrep.simx_opmode_buffer)

        return np.concatenate([state, gripper_joint_vel, gripper_joint_angles])

    def _loadModelsAndAssignHandles(self):
        # get handles
        super()._loadModelsAndAssignHandles()
        _, self.gripper_f1_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
        _, self.gripper_f2_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)

    def _setupDatastream(self):
        # set up datastreams
        super()._setupDatastream()
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f1_handle, 2012, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f1_handle, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f2_handle, 2012, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f2_handle, vrep.simx_opmode_streaming)


    def reset(self):
        """
        Reset the environment
        Return initial state and assign the internal initial state
        Close the gripper hand
        """
        current_state = super().reset()
        for _ in range(int(self.__class__.RESET_TIME/VREPPushTaskEnvironment.SIMULATION_DT)):
            vrep.simxPauseCommunication(self.client_ID, 1)
            for j in range(6):
                vrep.simxSetJointTargetVelocity(self.client_ID, self.joint_handles[j], 0.0,
                        vrep.simx_opmode_oneshot)
            # -1 == open, 1 == close
            vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f1_handle, self._gripper_closing_vel,
                    vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f2_handle, self._gripper_closing_vel,
                    vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.client_ID, 0)
            vrep.simxSynchronousTrigger(self.client_ID)
            # make sure all commands are exeucted
            vrep.simxGetPingTime(self.client_ID)

        # obtain the first state after the first time step; it injects some noise into the distribution of the initial state
        current_state = self.getCurrentState(self.client_ID, self.joint_handles, self.gripper_handle, self.cuboid_handle,
                self.target_plane_handle)

        self.state = current_state
        return current_state


    def step(self, actions):
        """
        Execute sequences of actions (None, 7) in the environment
        Return sequences of subsequent states and rewards
        """
        next_states = []
        rewards = []
        for i in range(actions.shape[0]):
            current_vel = self.state[:6] + actions[i, :6]
            vrep.simxPauseCommunication(self.client_ID, 1)
            for j in range(6):
                # Cap at max velocity
                vel = max(-VREPPushTaskEnvironment.MAX_JOINT_VELOCITY, min(VREPPushTaskEnvironment.MAX_JOINT_VELOCITY,
                    current_vel[j]))
                vrep.simxSetJointTargetVelocity(self.client_ID, self.joint_handles[j], vel,
                        vrep.simx_opmode_oneshot)
            # -1 == open, 1 == close
            gripper_vel = (actions[i, 6]/np.abs(actions[i, 6])) * self._gripper_closing_vel

            vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f1_handle, gripper_vel,
                    vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f2_handle, gripper_vel,
                    vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.client_ID, 0)
            vrep.simxSynchronousTrigger(self.client_ID)
            # make sure all commands are exeucted
            vrep.simxGetPingTime(self.client_ID)
            # obtain next state
            next_state = self.getCurrentState(self.client_ID, self.joint_handles, self.gripper_handle, self.cuboid_handle,
                    self.target_plane_handle)
            next_states.append(next_state)
            rewards.append(VREPPushTaskEnvironment.getRewards(self.state[:24], actions[i, :6]))
            self.state = np.copy(next_state)

        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        return next_states, rewards, False, None


def make(env_name, *args, **kwargs):
    if env_name == "VREPPushTask":
        return VREPPushTaskEnvironment(*args, **kwargs)
    elif env_name == "VREPPushTask2":
        return VREPPushTaskEnvironment(
                *args,
                **kwargs,
                init_cb_pos=[0.3, -0.5, 0.05],
                init_tg_pos=[0.3, -0.8, 0.002],
                )
    elif env_name == "VREPPushTask3":
        return VREPPushTaskEnvironment(
                *args,
                **kwargs,
                init_cb_pos=[0.55, 0., 0.05],
                init_tg_pos=[0.8, 0., 0.002],
                )
    elif env_name == "VREPPushTask4":
        return VREPPushTaskEnvironment(
                *args,
                **kwargs,
                init_cb_pos=[-0.55, 0., 0.05],
                init_tg_pos=[-0.8, 0., 0.002],
                )
    if env_name == "VREPPushTaskNonIK":
        return VREPPushTaskNonIKEnvironment(
                *args,
                **kwargs,
                mico_model_path="models/robots/non-mobile/MicoRobotNonIK.ttm")
    elif env_name == "VREPPushTaskContact":
        return VREPPushTaskEnvironment(
                *args,
                **kwargs,
                init_joint_pos=[np.pi, 5.0, np.pi, np.pi, np.pi, 3.40],
                init_cb_pos=[0.35, 0.35, 0.05],
                )
    elif env_name == "VREPPushTaskContact2":
        return VREPPushTaskEnvironment(
                *args,
                **kwargs,
                init_joint_pos=[np.pi, 5.0, np.pi, np.pi+0.1, np.pi, 3.40],
                init_cb_pos=[0.33, 0.35, 0.05],
                init_cb_orient=[0., 0., 0.5],
                init_tg_pos=[0.1, 0.7, 0.002],
                )
    elif env_name == "VREPPushTaskMultiStepReward":
        return VREPPushTaskMultiStepRewardEnvironment(
                *args,
                **kwargs,
                )
    elif env_name == "VREPPushTaskMultiStepRewardContact2":
        return VREPPushTaskMultiStepRewardEnvironment(
                *args,
                **kwargs,
                init_joint_pos=[np.pi, 5.0, np.pi, np.pi+0.1, np.pi, 3.40],
                init_cb_pos=[0.33, 0.35, 0.05],
                init_cb_orient=[0., 0., 0.5],
                init_tg_pos=[0.1, 0.7, 0.002],
                )
    else:
        raise IOError("Invalid VREP Environment name")
