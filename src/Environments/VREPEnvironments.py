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

class VREPEnvironment(object):
    SIMULATION_DT = 0.05
    MAX_JOINT_VELOCITY_DELTA = 1.0
    MAX_JOINT_VELOCITY = 6.0

    def __init__(self, port=19997):
        logger.info("Creating {}".format(self.__class__.__name__))
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.client_ID=vrep.simxStart('127.0.0.1',port,True,True,5000,5) # Connect to V-REP
        if self.client_ID == -1:
            raise IOError("VREP connection failed.")

        # enable the synchronous mode on the client:
        vrep.simxSynchronous(self.client_ID,True)
        # start the simulation:
        vrep.simxStartSimulation(self.client_ID, vrep.simx_opmode_blocking)
        self._reset_yet = False
        self._step = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        return False

    def close(self):
        logger.info("Closing {}".format(self.__class__.__name__))

        #self._tearDownDatastream()

        # stop the simulation:
        vrep.simxStopSimulation(self.client_ID, vrep.simx_opmode_blocking)
        # before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(self.client_ID)
        # disconnect
        vrep.simxFinish(self.client_ID)


class VREPPushTaskEnvironment(VREPEnvironment):
    """
    Distance unit: m
    Maximum distance between target and cuboid: 2m; this will affect the reward function for VREPPushTaskMultiStepRewardEnvironment
    """
    CUBOID_SIDE_LENGTH = 0.1
    GRIPPER_BASE_TO_CLOSED_TIP_DIST = 0.15
    DEFAULT_JOINT_POSITIONS = [np.pi, 1.5 * np.pi, 1.5 * np.pi, np.pi, np.pi, np.pi]
    DEFAULT_CUBOID_POSITION = [0.3, 0.5, 0.05]
    DEFAULT_CUBOID_ORIENTATION = [0., 0., 0.]
    DEFAULT_TARGET_POSITION = [0.3, 0.8, 0.002]
    # Simulation delta time in seconds
    MAX_STEP = 100
    action_space = Box((6,), (-1.0,), (1.0,))
    observation_space = Box((24,), (-999.0,), (999.0,))

    def __init__(self, port=19997, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None,
                mico_model_path="models/robots/non-mobile/MicoRobot.ttm"):
        super().__init__(port)
        self._init_joint_pos = init_joint_pos if not init_joint_pos is None else VREPPushTaskEnvironment.DEFAULT_JOINT_POSITIONS
        self._init_cb_pos = init_cb_pos if not init_cb_pos is None else VREPPushTaskEnvironment.DEFAULT_CUBOID_POSITION
        self._init_cb_orient = init_cb_orient if not init_cb_orient is None else\
                                VREPPushTaskEnvironment.DEFAULT_CUBOID_ORIENTATION
        self._init_tg_pos = init_tg_pos if not init_tg_pos is None else VREPPushTaskEnvironment.DEFAULT_TARGET_POSITION
        self.mico_model_path = mico_model_path

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

    def getStateString(self, state):
        """
        Return a formatted string representation of the state

        Args
        -------
        state:              array(state_dim)/array(1, state_dim)

        Returns
        -------
        state_str:          String      Formatted string
        """
        state = state.flatten()
        assert(state.shape[0] == VREPPushTaskEnvironment.observation_space.shape[0])
        state_str = "joint_vel: {}\njoint_angles: {}\ngripper_pos: {}\ngripper_orient: {}\n"
        state_str += "cuboid_gripper_vec: {}\ntarget_plane_cuboid_vec: {}\n"
        return state_str.format(state[:6], state[6:12], state[12:15], state[15:18],
                state[18:21], state[21:24])

    def getCurrentState(self, client_ID, joint_handles, gripper_handle, cuboid_handle, target_plane_handle):
        """
            TODO: Refactor away arguments
            Return the state as an array of shape (24, )
            [current_vel, joint_angles, gripper_pos, gripper_orient, cuboid_gripper_vec, target_plane_cuboid_vec]
             6              6               3           3               3                   3
        """
        #print("PARENT 1")
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
        #print("PARENT 2")
        ret, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        #print("PARENT 3")
        ret, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_buffer)
        gripper_pos = np.array(gripper_pos)
        gripper_orient = np.array(gripper_orient)

        #print("PARENT 4")
        ret, cuboid_pos = vrep.simxGetObjectPosition(client_ID, cuboid_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, cuboid_pos = vrep.simxGetObjectPosition(client_ID, cuboid_handle, -1, vrep.simx_opmode_buffer)
        cuboid_pos = np.array(cuboid_pos)

        #print("PARENT 5")
        ret, target_plane_pos = vrep.simxGetObjectPosition(client_ID, target_plane_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, target_plane_pos = vrep.simxGetObjectPosition(client_ID, target_plane_handle, -1, vrep.simx_opmode_buffer)
        target_plane_pos = np.array(target_plane_pos)

        #print("PARENT 6")
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
        self._step = 0

        return current_state

    def getRewardsDense(self, state, action, next_state):
        """
            Return the sum of the Euclidean distance between gripper and cuboid and the Euclidean distance between cuboid and targetPlane.
            Args
            -------
            state:   array(state_dim)/array(batch_size, state_dim)
            action:  array(action_dim)/array(batch_size, action_dim)
            next_state:   array(state_dim)/array(batch_size, state_dim)

            Returns
            -------
            reward:  array(batch_size, 1)

            NB: Rewards should be non-negative
        """
        state = state.reshape(-1, VREPPushTaskEnvironment.observation_space.shape[0])
        batch_size = state.shape[0]
        action = action.reshape(batch_size, -1)
        next_state = next_state.reshape(batch_size, -1)
        return (-(np.sqrt(np.sum(np.square(state[:, -6:-3]), axis=1)) + np.sqrt(np.sum(np.square(state[:, -3:]),
            axis=1)))).reshape(batch_size, 1)
        #return -(np.sqrt(np.sum(np.square(action))))
        #return np.tanh(-(np.sqrt(np.sum(np.square(state[:1]))))/10.0) + 1.0

    def getRewards(self, state, action, next_state):
        return self.getRewardsDense(state, action, next_state)

    def _reachedGoalState(self, state):
        """
        If state has reached goal state
        """
        cube_to_target_dist = np.sqrt(np.sum(np.square(state[:, 21:24]), axis=1))
        return (cube_to_target_dist <= self.CUBOID_SIDE_LENGTH / 2 + 0.1).squeeze()

    def _isDone(self):
        """
        Criteria for termination
        """
        state = self.state.reshape(1, -1)
        assert(state.shape[1] == self.observation_space.shape[0])
        return self._reachedGoalState(state) or self._step >= self.MAX_STEP

    def step(self, actions):
        """
        Execute sequences of actions (None, action_dim) in the environment
        Return sequences of subsequent states and rewards

        Args
        -------
        actions:  array(-1, action_dim)

        Returns
        -------
        next_states:   array(-1, state_dim)
        rewards:    array(-1, 1)
        done:   Boolean
        info:   None
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
            next_states.append(next_state.reshape(1, -1))
            rewards.append(self.getRewards(self.state, actions[i], next_state))
            self._step += 1
            self.state = np.copy(next_state)
            if self._isDone():
                break

        next_states = np.concatenate(next_states, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        return next_states, rewards, self._isDone(), None


class VREPPushTaskMultiStepRewardEnvironment(VREPPushTaskEnvironment):

    def __init__(self, port=19997, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None):
        super().__init__(port, init_joint_pos, init_cb_pos, init_cb_orient, init_tg_pos)

    def getRewards(self, state, action, next_state):
        gripper_cube_dist = np.sqrt(np.sum(np.square(state[-6:-3])))
        if gripper_cube_dist >= 0.1 + VREPPushTaskEnvironment.GRIPPER_BASE_TO_CLOSED_TIP_DIST + (VREPPushTaskEnvironment.CUBOID_SIDE_LENGTH * np.sqrt(2) / 2):
            return -gripper_cube_dist
        return -gripper_cube_dist + 1 - np.sqrt(np.sum(np.square(state[-3:])))

class VREPPushTask7DoFEnvironment(VREPPushTaskEnvironment):
    """
    Distance unit: m
    Maximum distance between target and cuboid: 2m; this will affect the reward function for VREPPushTaskMultiStepRewardEnvironment
    """
    # Reset time in seconds
    RESET_TIME = 1.2
    action_space = Box((7,), (-1.0,), (1.0,))
    observation_space = Box((28,), (-5.0,), (5.0,))
    goal_space = Box((6,), (-5.0,), (5.0,))
    reward_space = Box((1,), (-5.0,), (5.0,))

    def __init__(self, port=19997, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None,
                mico_model_path="models/robots/non-mobile/MicoRobot7DoF.ttm", goal=None):
        super().__init__(port, init_joint_pos, init_cb_pos, init_cb_orient, init_tg_pos, mico_model_path)
        self._gripper_closing_vel = -0.04
        # goal:   array(6,) [goal_gripper_cube_vec, goal_cube_target_vec]
        # Default goal is to have the gripper reach the cube and have the cube hit the target
        self._goal = np.zeros(self.goal_space.shape[0]) if goal == None else goal

    def _tearDownDatastream(self):
        # tear down datastreams
        super()._tearDownDatastream()
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f1_handle, 2012,
                vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f1_handle, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f2_handle, 2012,
                vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f2_handle, vrep.simx_opmode_discontinue)

    def extractGoal(self, state):
        state = state.reshape((-1, VREPPushTask7DoFEnvironment.observation_space.shape[0]))
        return state[:, 18:24]


    def getRewardsDense(self, state, action, next_state, goal=None):
        """
            Return the sum of the Euclidean distance between gripper and cuboid and the Euclidean distance between cuboid and targetPlane.
            Args
            -------
            state:   array(state_dim)/array(batch_size, state_dim)
            action:  array(action_dim)/array(batch_size, action_dim)
            next_state:   array(state_dim)/array(batch_size, state_dim)
            *goal:   array(6,)/array(batch_size, 6,) [goal_gripper_cube_vec, goal_cube_target_vec]

            Returns
            -------
            reward:  array(batch_size, 1)

            NB: Rewards should be non-negative
        """
        state = state.reshape(-1, VREPPushTask7DoFEnvironment.observation_space.shape[0])
        batch_size = state.shape[0]
        action = action.reshape(batch_size, -1)
        next_state = next_state.reshape(batch_size, -1)
        state = state[:, :24]
        action = action[:, :6]
        next_state = next_state[:, :24]
        if goal is None:
            goal = np.array([self._goal for _ in range(batch_size)])
        goal = goal.reshape((batch_size, 6))
        reward = (-(np.sqrt(np.sum(np.square(state[:, -6:-3] - goal[:, -6:-3]), axis=1)) + \
                np.sqrt(np.sum(np.square(state[:, -3:] - goal[:, -3:]), axis=1)))).reshape(batch_size, 1)
        return np.clip(reward, self.reward_space.low[0], self.reward_space.high[0])
        #return -(np.sqrt(np.sum(np.square(action))))
        #return np.tanh(-(np.sqrt(np.sum(np.square(state[:1]))))/10.0) + 1.0

    def getRewards(self, state, action, next_state, goal=None):
        return self.getRewardsDense(state, action, next_state, goal)


    def getStateString(self, state):
        """
        Return a formatted string representation of the state

        Args
        -------
        state:              array(state_dim)/array(1, state_dim)

        Returns
        -------
        state_str:          String      Formatted string
        """
        state = state.flatten()
        assert(state.shape[0] == VREPPushTask7DoFEnvironment.observation_space.shape[0])
        state_str = super().getStateString(state[:24])
        state_str += "gripper_joint_vel: {}\ngripper_joint_angles: {}\n".format(
                state[24:26], state[26:28])
        return state_str

    def getCurrentState(self, client_ID, joint_handles, gripper_handle, cuboid_handle, target_plane_handle):
        """
            TODO: Refactor away arguments
            Return the state as an array of shape (28, )
            [joint_vel, joint_angles, gripper_pos, gripper_orient, cuboid_gripper_vec, target_plane_cuboid_vec,
             gripper_joint_vel,  gripper_joint_angles, ]
             6              6               3           3               3                   3
             2                      2
        """
        #print("GET CURRENT STATE1")
        state = super().getCurrentState(client_ID, joint_handles, gripper_handle, cuboid_handle, target_plane_handle)
        # NOTE: clip cb_grp_vec and tg_cb_vec
        state[18:24] = np.clip(state[18:24], self.observation_space.low[0], self.observation_space.high[0])
        #print("GET CURRENT STATE2")
        # Gripper joints
        gripper_joint_vel = np.array([0.0, 0.0], dtype='float')
        gripper_joint_angles = np.array([0.0, 0.0], dtype='float')

        #print("GET CURRENT STATE3")
        ret, gripper_joint_vel[0] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f1_handle, 2012,
                vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_vel[0] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f1_handle, 2012,
                    vrep.simx_opmode_buffer)
        #print("GET CURRENT STATE4")
        ret, gripper_joint_vel[1] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f2_handle, 2012,
                vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_vel[1] = vrep.simxGetObjectFloatParameter(client_ID, self.gripper_f2_handle, 2012,
                    vrep.simx_opmode_buffer)
        #print("GET CURRENT STATE5")
        ret, gripper_joint_angles[0] = vrep.simxGetJointPosition(client_ID, self.gripper_f1_handle, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_angles[0] = vrep.simxGetJointPosition(client_ID, self.gripper_f1_handle, vrep.simx_opmode_buffer)
        #print("GET CURRENT STATE6")
        ret, gripper_joint_angles[1] = vrep.simxGetJointPosition(client_ID, self.gripper_f2_handle, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_angles[1] = vrep.simxGetJointPosition(client_ID, self.gripper_f2_handle, vrep.simx_opmode_buffer)
        #print("GET CURRENT STATE7")

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
        for _ in range(int(self.RESET_TIME/self.SIMULATION_DT)):
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

        Args
        -------
        actions:  array(-1, 7)

        Returns
        -------
        next_states:   array(-1, 28)
        rewards:    array(-1, 1)
        done:   Boolean
        info:   None
        """
        next_states = []
        rewards = []
        rewards_dense = []
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
            next_states.append(next_state.reshape(1, -1))
            rewards.append(self.getRewards(self.state, actions, next_state))
            rewards_dense.append(self.getRewardsDense(self.state, actions, next_state))
            self._step += 1
            self.state = np.copy(next_state)
            if self._isDone():
                break

        next_states = np.concatenate(next_states, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        rewards_dense = np.concatenate(rewards_dense, axis=0)
        return next_states, rewards, rewards_dense, self._isDone(), None

class VREPPushTask7DoFIKEnvironment(VREPPushTask7DoFEnvironment):
    """
    Distance unit: m
    Maximum distance between target and cuboid: 2m; this will affect the reward function for VREPPushTaskMultiStepRewardEnvironment
    """
    # Reset time in seconds
    action_space = Box((11,), (-999.0,), (999.0,))
    #observation_space = Box((28,), (-999.0,), (999.0,))

    def __init__(self, port=19997, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None,
                mico_model_path="models/robots/non-mobile/MicoRobotIK.ttm"):
        super().__init__(port, init_joint_pos, init_cb_pos, init_cb_orient, init_tg_pos, mico_model_path)

    def _loadModelsAndAssignHandles(self):
        # get handles
        super()._loadModelsAndAssignHandles()
        _, self.gripper_tt_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand_Tip_Target_Sphere',
                vrep.simx_opmode_blocking)

    def _initialiseScene(self):
        """
            Initiailise gripper tip target to be at the same pos and orient as the gripper

        """
        #print(11)
        super()._initialiseScene()
        #print(12)
        vrep.simxGetPingTime(self.client_ID)
        _, self._gripper_tt_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1,
                vrep.simx_opmode_blocking)
        _, self._gripper_tt_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1,
                vrep.simx_opmode_blocking)
        #print(13)
        vrep.simxPauseCommunication(self.client_ID, 1)
        ret = vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_pos, vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_orient, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.client_ID, 0)
        #print(14)

    def step(self, actions):
        """
        Execute sequences of actions (None, action_dim) in the environment
        Return sequences of subsequent states and rewards

        Args
        -------
        actions:  array(-1, action_dim)
                  [tip_tgt_x, tip_tgt_y, tip_tgt_z, tip_tgt_rot_x, tip_tgt_rot_y, tip_tgt_rot_z,
                  tip_tgt_rot_use_world_frame, tip_tgt_rot_fixed, gripper_close, step_action, reset_tip_tgt]

        Returns
        -------
        next_states:   array(-1, state_dim)
        rewards:    array(-1, 1)
        done:   Boolean
        info:   None
        """
        next_states = []
        rewards = []
        for i in range(actions.shape[0]):
            # if not step action
            if actions[i, 9] < 0:
                # Actuate gripper tip target
                # if reset_tip_tgt
                if actions[i, 10] > 0:
                    vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_pos,
                                vrep.simx_opmode_blocking)
                    vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_orient,
                            vrep.simx_opmode_blocking)
                else:
                    # Algin gripper tip target orientation with gripper orientation if necessary
                    _, gripper_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1,
                            vrep.simx_opmode_buffer)
                    _, gripper_tt_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, vrep.simx_opmode_blocking)
                    dist = np.sqrt(np.sum(np.square(np.array(gripper_orient) - np.array(gripper_tt_orient))))
                    # if not tip_tgt_rot_fixed
                    if actions[i, 7] < 0 and dist > 0.3:
                        vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, gripper_orient,
                                vrep.simx_opmode_blocking)
                    # Apply action
                    # Actuate tip target position
                    # if tip_tgt_rot_use_world_frame
                    if actions[i, 6] > 0:
                        _, gripper_tt_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_tt_handle, -1,
                                vrep.simx_opmode_blocking)
                        newPos = np.array(gripper_tt_pos) + actions[i, :3]
                        vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, -1, list(newPos),
                                vrep.simx_opmode_blocking)
                    else:
                        vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, self.gripper_tt_handle,
                                list(actions[i, :3]), vrep.simx_opmode_blocking)
                    # Actuate tip target orientation
                    vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, self.gripper_tt_handle,
                            list(actions[i, 3:6]), vrep.simx_opmode_blocking)

                # Actuate gripper
                # -1 == open, 1 == close
                gripper_vel = (actions[i, 8]/np.abs(actions[i, 8])) * self._gripper_closing_vel

                vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f1_handle, gripper_vel,
                        vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f2_handle, gripper_vel,
                        vrep.simx_opmode_blocking)
            else:
                # Real step
                # Update gripper_tt pos and orients to current GRIPPER (NOT Gripper TT) pos and orient (for tt reset)
                #print("step0")
                _, self._gripper_tt_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1,
                        vrep.simx_opmode_buffer)
                _, self._gripper_tt_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1,
                        vrep.simx_opmode_buffer)
                #print("step1")
                vrep.simxSynchronousTrigger(self.client_ID)
                #print("step11")
                # make sure all commands are exeucted
                vrep.simxGetPingTime(self.client_ID)
                # obtain next state
                #print("step12")
                next_state = self.getCurrentState(self.client_ID, self.joint_handles, self.gripper_handle, self.cuboid_handle,
                        self.target_plane_handle)
                #print("step13")
                next_states.append(next_state.reshape(1, -1))
                # NOTE: actions is not relevant in calculating rewards
                rewards.append(self.getRewards(self.state, actions[:, :7], next_state))
                #print("step14")
                #print("step2")
                self._step += 1
                self.state = np.copy(next_state)
                if self._isDone():
                    break

        if len(next_states) > 0:
            next_states = np.concatenate(next_states, axis=0)
        else:
            next_states = None
        if len(rewards) > 0:
            rewards = np.concatenate(rewards, axis=0)
        else:
            next_states = None
        return next_states, rewards, rewards, self._isDone(), None


class VREPPushTask7DoFSparseRewardsEnvironment(VREPPushTask7DoFEnvironment):

    def __init__(self, port=19997, init_joint_pos=None, init_cb_pos=None, init_cb_orient=None, init_tg_pos=None,
                mico_model_path="models/robots/non-mobile/MicoRobot7DoF.ttm"):
        super().__init__(port, init_joint_pos, init_cb_pos, init_cb_orient, init_tg_pos, mico_model_path)


    def getRewards(self, state, action, next_state, goal=None):
        """
        Sparse Rewards. Big reward at the end of the episode.
        Args
        -------
        state:   array(state_dim)/array(batch_size, state_dim)
        action:  array(action_dim)/array(batch_size, action_dim)
        *goal:   array(6,)/array(batch_size, 6,) [goal_gripper_cube_vec, goal_cube_target_vec]

        Returns
        -------
        reward:  array(batch_size, 1)

        NOTE: Rewards should be non-negative
        NOTE: The reward is calculated in terms of the next_state instead of current state.
        """
        state = state.reshape(-1, VREPPushTask7DoFEnvironment.observation_space.shape[0])
        batch_size = state.shape[0]
        action = action.reshape(batch_size, -1)
        next_state = next_state.reshape(batch_size, -1)
        if goal is None:
            goal = np.array([self._goal for _ in range(batch_size)])
        goal = goal.reshape((batch_size, 6))
        cube_to_goal_dists = np.sqrt(np.sum(np.square(next_state[:, 21:24] - goal[:, -3:]), axis=1))
        reward = np.zeros_like(cube_to_goal_dists) - 0.01
        reward[(cube_to_goal_dists <= self.CUBOID_SIDE_LENGTH / 2 + 0.1).squeeze()] = 1.0
        reward = reward.reshape((-1, 1))
        assert reward.shape[0] == batch_size
        return reward


class VREPPushTask7DoFSparseRewardsIKEnvironment(VREPPushTask7DoFSparseRewardsEnvironment,
        VREPPushTask7DoFIKEnvironment):
    pass


class VREPGraspTask7DoFEnvironment(VREPEnvironment):
    DEFAULT_CUP_POSITION = [0.0, 0.5, 0.22]
    DEFAULT_CUP_ORIENTATION = [0., 0., 0.]
    DEFAULT_TARGET_CUP_POSITION = [0.0, 0.5, 0.35]
    DEFAULT_TARGET_CUP_ORIENTATION = [0., -np.pi/2., 0.]
    MAX_STEP = 400
    action_space = Box((7,), (-1.0,), (1.0,))
    observation_space = Box((40,), (-5.0,), (5.0,))
    goal_space = Box((6,), (-5.0,), (5.0,))
    reward_space = Box((1,), (-5.0,), (5.0,))

    def __init__(self, port=19997, init_cup_pos=None, init_cup_orient=None, init_tg_cup_pos=None,
            init_tg_cup_orient=None,
                mico_model_path="models/robots/non-mobile/MicoRobot7DoF2.ttm", goal=None):
        super().__init__(port)
        self._init_cup_pos = init_cup_pos if not init_cup_pos is None else\
                                VREPGraspTask7DoFEnvironment.DEFAULT_CUP_POSITION
        self._init_cup_orient = init_cup_orient if not init_cup_orient is None else\
                                VREPGraspTask7DoFEnvironment.DEFAULT_CUP_ORIENTATION
        self._init_tg_cup_pos = init_tg_cup_pos if not init_tg_cup_pos is None else\
                                VREPGraspTask7DoFEnvironment.DEFAULT_TARGET_CUP_POSITION
        self._init_tg_cup_orient = init_tg_cup_orient if not init_tg_cup_orient is None else\
                                VREPGraspTask7DoFEnvironment.DEFAULT_TARGET_CUP_ORIENTATION
        self.mico_model_path = mico_model_path
        self._gripper_closing_vel = -0.04
        # *goal = array(6,) [mean_grp_spots_vec, mean_spots_targets_vec]
        self._goal = np.zeros(self.goal_space.shape[0]) if goal == None else goal

    def _tearDownDatastream(self):
        # tear down datastreams
        for i in range(6):
            _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.joint_handles[i], 2012, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetJointPosition(self.client_ID, self.joint_handles[i],
                    vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.spot_bot_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.spot_r_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.spot_l_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_bot_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_r_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_l_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f1_handle, 2012,
                vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f1_handle, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f2_handle, 2012,
                vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f2_handle, vrep.simx_opmode_discontinue)

    def getStateString(self, state):
        """
        Return a formatted string representation of the state

        Args
        -------
        state:              array(state_dim)/array(1, state_dim)

        Returns
        -------
        state_str:          String      Formatted string
        """
        state = state.flatten()
        assert(state.shape[0] == VREPGraspTask7DoFEnvironment.observation_space.shape[0])
        state_str = "joint_vel: {}\njoint_angles: {}\ngripper_pos: {}\ngripper_orient: {}\n"
        state_str += "spot_bot_gripper_vec: {}\nspot_l_gripper_vec: {}\nspot_r_gripper_vec: {}\n"
        state_str += "tg_spot_bot_spot_bot_vec: {}\ntg_spot_l_spot_l_vec: {}\ntg_spot_r_spot_r_vec: {}\n"
        state_str += "gripper_joint_vel: {}\ngripper_joint_angles: {}\n"
        return state_str.format(state[:6], state[6:12], state[12:15], state[15:18],
                state[18:21], state[21:24], state[24:27],
                state[27:30], state[30:33], state[33:36],
                state[36:38], state[38:40])

    def getCurrentState(self):
        """
            TODO: Refactor away arguments
            Return the state as an array of shape (40, )

            0-6:        joint_vel
            6-12:       joint_pos
            12-18:      gripper_pose
            18-27:      spots to gripper vecs(3): bot, l, r
            27-36:      target to spots vecs(3): bot, l, r
            36-40:      gripper_vel + gripper_pos
        """
        #print("PARENT 1")
        current_vel = np.array([0, 0, 0, 0, 0, 0], dtype='float')
        joint_angles = np.array([0, 0, 0, 0, 0, 0], dtype='float')
        # obtain first state
        for i in range(6):
            ret, current_vel[i] = vrep.simxGetObjectFloatParameter(self.client_ID, self.joint_handles[i], 2012,
                    vrep.simx_opmode_buffer)
            while ret != vrep.simx_return_ok:
                ret, current_vel[i] = vrep.simxGetObjectFloatParameter(self.client_ID, self.joint_handles[i], 2012,
                        vrep.simx_opmode_buffer)
            ret, joint_angles[i] = vrep.simxGetJointPosition(self.client_ID, self.joint_handles[i], vrep.simx_opmode_buffer)
            while ret != vrep.simx_return_ok:
                ret, joint_angles[i] = vrep.simxGetJointPosition(self.client_ID, self.joint_handles[i], vrep.simx_opmode_buffer)
        #print("PARENT 2")
        ret, gripper_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_buffer)
        #print("PARENT 3")
        ret, gripper_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_buffer)
        gripper_pos = np.array(gripper_pos)
        gripper_orient = np.array(gripper_orient)

        # spots to gripper vec
        # NOTE: clipped
        #print("PARENT 4")
        ret, spot_bot_pos = vrep.simxGetObjectPosition(self.client_ID, self.spot_bot_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, spot_bot_pos = vrep.simxGetObjectPosition(self.client_ID, self.spot_bot_handle, -1, vrep.simx_opmode_buffer)
        spot_bot_pos = np.array(spot_bot_pos)

        ret, spot_l_pos = vrep.simxGetObjectPosition(self.client_ID, self.spot_l_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, spot_l_pos = vrep.simxGetObjectPosition(self.client_ID, self.spot_l_handle, -1, vrep.simx_opmode_buffer)
        spot_l_pos = np.array(spot_l_pos)

        ret, spot_r_pos = vrep.simxGetObjectPosition(self.client_ID, self.spot_r_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, spot_r_pos = vrep.simxGetObjectPosition(self.client_ID, self.spot_r_handle, -1, vrep.simx_opmode_buffer)
        spot_r_pos = np.array(spot_r_pos)

        hi, lo = VREPGraspTask7DoFEnvironment.observation_space.high[0],\
                    VREPGraspTask7DoFEnvironment.observation_space.low[0]
        spot_bot_gripper_vec = np.clip(spot_bot_pos - gripper_pos, lo, hi)
        spot_l_gripper_vec = np.clip(spot_l_pos - gripper_pos, lo, hi)
        spot_r_gripper_vec = np.clip(spot_r_pos - gripper_pos, lo, hi)

        # target spots to spots vec
        # NOTE: clipped
        #print("PARENT 5")
        ret, tg_spot_bot_pos = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_bot_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, tg_spot_bot_pos = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_bot_handle, -1, vrep.simx_opmode_buffer)
        tg_spot_bot_pos = np.array(tg_spot_bot_pos)

        ret, tg_spot_l_pos = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_l_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, tg_spot_l_pos = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_l_handle, -1, vrep.simx_opmode_buffer)
        tg_spot_l_pos = np.array(tg_spot_l_pos)

        ret, tg_spot_r_pos = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_r_handle, -1, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, tg_spot_r_pos = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_r_handle, -1, vrep.simx_opmode_buffer)
        tg_spot_r_pos = np.array(tg_spot_r_pos)

        #print("PARENT 6")
        tg_spot_bot_vec = np.clip(tg_spot_bot_pos - spot_bot_pos, lo, hi)
        tg_spot_l_vec = np.clip(tg_spot_l_pos - spot_l_pos, lo, hi)
        tg_spot_r_vec = np.clip(tg_spot_r_pos - spot_r_pos, lo, hi)

        # Gripper f1 and f2 vels and pos
        gripper_joint_vel = np.array([0.0, 0.0], dtype='float')
        gripper_joint_angles = np.array([0.0, 0.0], dtype='float')

        #print("GET CURRENT STATE3")
        ret, gripper_joint_vel[0] = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f1_handle, 2012,
                vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_vel[0] = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f1_handle, 2012,
                    vrep.simx_opmode_buffer)
        #print("GET CURRENT STATE4")
        ret, gripper_joint_vel[1] = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f2_handle, 2012,
                vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_vel[1] = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f2_handle, 2012,
                    vrep.simx_opmode_buffer)
        #print("GET CURRENT STATE5")
        ret, gripper_joint_angles[0] = vrep.simxGetJointPosition(self.client_ID, self.gripper_f1_handle, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_angles[0] = vrep.simxGetJointPosition(self.client_ID, self.gripper_f1_handle, vrep.simx_opmode_buffer)
        #print("GET CURRENT STATE6")
        ret, gripper_joint_angles[1] = vrep.simxGetJointPosition(self.client_ID, self.gripper_f2_handle, vrep.simx_opmode_buffer)
        while ret != vrep.simx_return_ok:
            ret, gripper_joint_angles[1] = vrep.simxGetJointPosition(self.client_ID, self.gripper_f2_handle, vrep.simx_opmode_buffer)

        return np.concatenate([current_vel, joint_angles, gripper_pos, gripper_orient,
            spot_bot_gripper_vec, spot_l_gripper_vec, spot_r_gripper_vec,
            tg_spot_bot_vec, tg_spot_l_vec, tg_spot_r_vec,
            gripper_joint_vel, gripper_joint_angles])

    def _loadModelsAndAssignHandles(self):
        # get handles
        # cup
        _, self.cup_handle = vrep.simxGetObjectHandle(self.client_ID, 'Cup', vrep.simx_opmode_blocking)
        # cup spots
        _, self.spot_bot_handle = vrep.simxGetObjectHandle(self.client_ID, 'SpotBot', vrep.simx_opmode_blocking)
        _, self.spot_l_handle = vrep.simxGetObjectHandle(self.client_ID, 'SpotL', vrep.simx_opmode_blocking)
        _, self.spot_r_handle = vrep.simxGetObjectHandle(self.client_ID, 'SpotR', vrep.simx_opmode_blocking)

        # target cup
        _, self.tg_cup_handle = vrep.simxGetObjectHandle(self.client_ID, 'CupTarget', vrep.simx_opmode_blocking)
        # target cup spots
        _, self.tg_spot_bot_handle = vrep.simxGetObjectHandle(self.client_ID, 'SpotBotTarget', vrep.simx_opmode_blocking)
        _, self.tg_spot_l_handle = vrep.simxGetObjectHandle(self.client_ID, 'SpotLTarget', vrep.simx_opmode_blocking)
        _, self.tg_spot_r_handle = vrep.simxGetObjectHandle(self.client_ID, 'SpotRTarget', vrep.simx_opmode_blocking)

        _, self.model_base_handle = vrep.simxLoadModel(self.client_ID, self.mico_model_path, 0, vrep.simx_opmode_blocking)
        self.joint_handles = [-1, -1, -1, -1, -1, -1]
        for i in range(6):
            _, self.joint_handles[i] = vrep.simxGetObjectHandle(self.client_ID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
        _, self.gripper_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand', vrep.simx_opmode_blocking)

        # gripper
        _, self.gripper_f1_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
        _, self.gripper_f2_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)


    def _initialiseScene(self):
        """
        Initialise the environment
        """
        # initialise cup orientation, cup position, target cup orientation and target cup position
        vrep.simxPauseCommunication(self.client_ID, 1)
        ret = vrep.simxSetObjectOrientation(self.client_ID, self.cup_handle, -1, self._init_cup_orient, vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectPosition(self.client_ID, self.cup_handle, -1, self._init_cup_pos, vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectOrientation(self.client_ID, self.tg_cup_handle, -1, self._init_tg_cup_orient, vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectPosition(self.client_ID, self.tg_cup_handle, -1, self._init_tg_cup_pos, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.client_ID, 0)

    def _setupDatastream(self):
        # set up datastreams
        current_vel = np.array([0, 0, 0, 0, 0, 0], dtype='float')
        joint_angles = np.array([0, 0, 0, 0, 0, 0], dtype='float')

        # set up datastreams
        for i in range(6):
            _, current_vel[i] = vrep.simxGetObjectFloatParameter(self.client_ID, self.joint_handles[i], 2012,
                    vrep.simx_opmode_streaming)
            _, joint_angles[i] = vrep.simxGetJointPosition(self.client_ID, self.joint_handles[i], vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_streaming)

        # Spots
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.spot_bot_handle, -1, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.spot_l_handle, -1, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.spot_r_handle, -1, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_bot_handle, -1, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_l_handle, -1, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectPosition(self.client_ID, self.tg_spot_r_handle, -1, vrep.simx_opmode_streaming)

        # Gripper joints
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f1_handle, 2012, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f1_handle, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.gripper_f2_handle, 2012, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetJointPosition(self.client_ID, self.gripper_f2_handle, vrep.simx_opmode_streaming)

        del current_vel, joint_angles

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
        current_state = self.getCurrentState()

        self.state = current_state
        self._step = 0

        return current_state

    def step(self, actions, vels_only=False):
        """
        Execute sequences of actions (None, 7) in the environment
        Return sequences of subsequent states and rewards
        1 = close; -1 = open

        Args
        -------
        actions:  array(-1, 7)

        Returns
        -------
        next_states:   array(-1, 40)
        rewards:    array(-1, 1)
        done:   Boolean
        info:   None
        """
        next_states = []
        rewards = []
        rewards_dense = []
        corrected_actions = np.copy(actions)
        for i in range(actions.shape[0]):
            if vels_only:
                current_vel = actions[i, :6]
                corrected_actions[i, :6] = current_vel - self.state[:6]

                #actions[i, :6] = current_vel - self.state[:6]
            else:
                current_vel = self.state[:6] + actions[i, :6]
            #print("Target vel")
            #print(current_vel)
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
            next_state = self.getCurrentState()
            next_states.append(next_state.reshape(1, -1))
            rewards.append(self.getRewards(self.state, actions, next_state))
            rewards_dense.append(self.getRewardsDense(self.state, actions, next_state))
            self._step += 1
            self.state = np.copy(next_state)
            if self._isDone():
                break

        next_states = np.concatenate(next_states, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        rewards_dense = np.concatenate(rewards_dense, axis=0)
        if vels_only:
            return next_states, rewards, rewards_dense, self._isDone(), None, corrected_actions
        return next_states, rewards, rewards_dense, self._isDone(), None

    def extractGoal(self, state):
        state = state.reshape((-1, VREPGraspTask7DoFEnvironment.observation_space.shape[0]))
        grp_spot_b = state[:, 18:21]
        grp_spot_l = state[:, 21:24]
        grp_spot_r = state[:, 24:27]
        spot_tg_b = state[:, 27:30]
        spot_tg_l = state[:, 30:33]
        spot_tg_r = state[:, 33:36]

        grp_spot = np.mean([grp_spot_b, grp_spot_l, grp_spot_r], axis=0)
        spot_tg = np.mean([spot_tg_b, spot_tg_l, spot_tg_r], axis=0)

        return np.concatenate([grp_spot, spot_tg], axis=1)

    def getRewardsDense(self, state, action, next_state, goal=None):
        """
        Sparse Rewards. Big reward at the end of the episode.
        Args
        -------
        state:   array(state_dim)/array(batch_size, state_dim)
        action:  array(action_dim)/array(batch_size, action_dim)

        Returns
        -------
        reward:  array(batch_size, 1)

        NOTE: The reward is calculated in terms of the next_state instead of current state.
        """
        state = state.reshape(-1, VREPGraspTask7DoFEnvironment.observation_space.shape[0])
        batch_size = state.shape[0]
        action = action.reshape(batch_size, -1)
        next_state = next_state.reshape(batch_size, -1)
        if goal is None:
            goal = np.array([self._goal for _ in range(batch_size)])
        goal = goal.reshape((batch_size, 6))

        gripper_spot_bot_dist = np.sqrt(np.sum(np.square(state[:, 18:21] - goal[:, -6:-3]), axis=1))
        gripper_spot_l_dist = np.sqrt(np.sum(np.square(state[:, 21:24] - goal[:, -6:-3]), axis=1))
        gripper_spot_r_dist = np.sqrt(np.sum(np.square(state[:, 24:27] - goal[:, -6:-3]), axis=1))
        tg_spot_bot_dist = np.sqrt(np.sum(np.square(state[:, 27:30] - goal[:, -3:]), axis=1))
        tg_spot_l_dist = np.sqrt(np.sum(np.square(state[:, 30:33] - goal[:, -3:]), axis=1))
        tg_spot_r_dist = np.sqrt(np.sum(np.square(state[:, 33:36] - goal[:, -3:]), axis=1))
        #print("{}, {}, {}".format(tg_spot_bot_dist, tg_spot_l_dist, tg_spot_r_dist))
        gripper_spot_dist = np.mean([gripper_spot_bot_dist, gripper_spot_l_dist, gripper_spot_r_dist], axis=0)
        tg_spot_dist = np.mean([tg_spot_bot_dist, tg_spot_l_dist, tg_spot_r_dist], axis=0)
        reward = -(gripper_spot_dist + tg_spot_dist)
        # Penalise when gripper is far from the cup and is closing
        #premature_closing_penalty = np.zeros_like(reward)
        #premature_closing_penalty[(gripper_spot_dist >= .15).squeeze() &
        #                        (action[:, 6] == 1).squeeze()] = -1

        #capture_reward = np.zeros_like(reward)
        ## Extra reward when gripper is very close to the cup and is closing
        #capture_reward[(gripper_spot_dist <= .12).squeeze() &
        #                        (action[:, 6] == 1).squeeze()] = 1

        #reward += premature_closing_penalty + capture_reward

        # clip reward to reward hi and reward lo:
        #reward[(reward > self.reward_space.high[0])] = self.reward_space.high[0]
        #reward[(reward < self.reward_space.low[0])] = self.reward_space.low[0]
        reward = np.clip(reward, self.reward_space.low[0], self.reward_space.high[0])
        reward = reward.reshape((-1, 1))
        assert reward.shape[0] == batch_size
        return reward

    def getRewards(self, state, action, next_state, goal=None):
        return self.getRewardsDense(state, action, next_state, goal)

    def _reachedGoalState(self, state, goal=None):
        """
        If state has reached goal state
        """
        state = state.reshape(-1, VREPGraspTask7DoFSparseRewardsEnvironment.observation_space.shape[0])
        batch_size = state.shape[0]
        if goal is None:
            goal = np.array([self._goal for _ in range(batch_size)])
        goal = goal.reshape((batch_size, 6))
        tg_spot_bot_dist = np.sqrt(np.sum(np.square(state[:, 27:30] - goal[:, -3:]), axis=1))
        tg_spot_l_dist = np.sqrt(np.sum(np.square(state[:, 30:33] - goal[:, -3:]), axis=1))
        tg_spot_r_dist = np.sqrt(np.sum(np.square(state[:, 33:36] - goal[:, -3:]), axis=1))
        #print("{}, {}, {}".format(tg_spot_bot_dist, tg_spot_l_dist, tg_spot_r_dist))
        min_dist = 0.06
        return ((tg_spot_bot_dist <= min_dist) & (tg_spot_l_dist <= min_dist) & (tg_spot_r_dist <= min_dist)).squeeze()

    def _isDone(self):
        state = self.state.reshape(1, -1)
        assert(state.shape[1] == self.observation_space.shape[0])
        return  self._reachedGoalState(state) or self._step >= self.MAX_STEP


class VREPGraspTask7DoFSparseRewardsEnvironment(VREPGraspTask7DoFEnvironment):
    def __init__(self, port=19997, init_cup_pos=None, init_cup_orient=None, init_tg_cup_pos=None,
            init_tg_cup_orient=None,
                mico_model_path="models/robots/non-mobile/MicoRobot7DoF2.ttm"):
        super().__init__(port, init_cup_pos, init_cup_orient, init_tg_cup_pos, init_tg_cup_orient,
                mico_model_path)

    def reset(self):
        current_state = super().reset()
        self._approached_cup = False
        return current_state

    def getRewards(self, state, action, next_state, approached_cup=None, goal=None):
        """
        Sparse Rewards. Big reward at the end of the episode.
        Args
        -------
        state:   array(state_dim)/array(batch_size, state_dim)
        action:  array(action_dim)/array(batch_size, action_dim)
        next_state:   array(state_dim)/array(batch_size, state_dim)
        approached_cup:          array(batch_size, 1)   NOTE: Only used in model-based

        Returns
        -------
        reward:  array(batch_size, 1)

        NOTE: The reward is calculated in terms of the next_state instead of current state.
        The reward has two stages:
            1. If the gripper base is within a certain distance to the three markers it obtains a ONE-TIME reward;
            2. AFTER the first reward is collected, if the three marksers are within a certain distance to their
            targets then the agent obtains the filnal reward.
        """
        state = state.reshape(-1, VREPGraspTask7DoFSparseRewardsEnvironment.observation_space.shape[0])
        batch_size = state.shape[0]
        action = action.reshape(batch_size, -1)
        next_state = next_state.reshape(batch_size, -1)

        gripper_spot_bot_dist = np.sqrt(np.sum(np.square(state[:, 18:21]), axis=1))
        gripper_spot_l_dist = np.sqrt(np.sum(np.square(state[:, 21:24]), axis=1))
        gripper_spot_r_dist = np.sqrt(np.sum(np.square(state[:, 24:27]), axis=1))
        #print("{}, {}, {}".format(tg_spot_bot_dist, tg_spot_l_dist, tg_spot_r_dist))
        min_dist = 0.15

        reward = np.zeros_like(gripper_spot_r_dist) - 0.2
        #approached_cup_persist = approached_cup if approached_cup is not None else self._approached_cup
        #approached_cup_persist = np.array(approached_cup_persist).flatten()

        #reward[approached_cup_persist & self._reachedGoalState(next_state)] = 10

        #approached = ((gripper_spot_bot_dist <= min_dist) | (gripper_spot_l_dist <= min_dist) | (gripper_spot_r_dist\
        #    <= min_dist)).squeeze()
        #reward[~approached_cup_persist & approached] = 5
        #approached_cup_persist = approached | approached_cup_persist

        reward[self._reachedGoalState(next_state, goal)] = 10
        reward = reward.reshape((-1, 1))
        assert reward.shape[0] == batch_size
        #if approached_cup is None:
        #    self._approached_cup = approached_cup_persist
        #    return reward
        #return reward, approached_cup_persist
        return reward

class VREPGraspTask7DoFSparseRewardsIKEnvironment(VREPGraspTask7DoFSparseRewardsEnvironment):
    """
    Distance unit: m
    Maximum distance between target and cuboid: 2m; this will affect the reward function for VREPPushTaskMultiStepRewardEnvironment
    """
    # Reset time in seconds
    action_space = Box((11,), (-999.0,), (999.0,))
    #observation_space = Box((28,), (-999.0,), (999.0,))

    def __init__(self, port=19997, init_cup_pos=None, init_cup_orient=None, init_tg_cup_pos=None,
            init_tg_cup_orient=None,
                mico_model_path="models/robots/non-mobile/MicoRobotIK.ttm"):
        super().__init__(port, init_cup_pos, init_cup_orient, init_tg_cup_pos, init_tg_cup_orient, mico_model_path)

    def _loadModelsAndAssignHandles(self):
        # get handles
        super()._loadModelsAndAssignHandles()
        _, self.gripper_tt_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand_Tip_Target_Sphere',
                vrep.simx_opmode_blocking)

    def _initialiseScene(self):
        """
            Initiailise gripper tip target to be at the same pos and orient as the gripper

        """
        #print(11)
        super()._initialiseScene()
        #print(12)
        vrep.simxGetPingTime(self.client_ID)
        _, self._gripper_tt_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1,
                vrep.simx_opmode_blocking)
        _, self._gripper_tt_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1,
                vrep.simx_opmode_blocking)
        #print(13)
        vrep.simxPauseCommunication(self.client_ID, 1)
        ret = vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_pos, vrep.simx_opmode_oneshot)
        ret = vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_orient, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.client_ID, 0)
        #print(14)

    def step(self, actions):
        """
        Execute sequences of actions (None, action_dim) in the environment
        Return sequences of subsequent states and rewards

        Args
        -------
        actions:  array(-1, action_dim)
                  [tip_tgt_x, tip_tgt_y, tip_tgt_z, tip_tgt_rot_x, tip_tgt_rot_y, tip_tgt_rot_z,
                  tip_tgt_rot_use_world_frame, tip_tgt_rot_fixed, gripper_close, step_action, reset_tip_tgt]

        Returns
        -------
        next_states:   array(-1, state_dim)
        rewards:    array(-1, 1)
        done:   Boolean
        info:   None
        """
        next_states = []
        rewards = []
        for i in range(actions.shape[0]):
            # if not step action
            if actions[i, 9] < 0:
                # Actuate gripper tip target
                # if reset_tip_tgt
                if actions[i, 10] > 0:
                    vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_pos,
                                vrep.simx_opmode_blocking)
                    vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, self._gripper_tt_orient,
                            vrep.simx_opmode_blocking)
                else:
                    # Algin gripper tip target orientation with gripper orientation if necessary
                    _, gripper_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1,
                            vrep.simx_opmode_buffer)
                    _, gripper_tt_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, vrep.simx_opmode_blocking)
                    dist = np.sqrt(np.sum(np.square(np.array(gripper_orient) - np.array(gripper_tt_orient))))
                    # if not tip_tgt_rot_fixed
                    if actions[i, 7] < 0 and dist > 0.3:
                        vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, -1, gripper_orient,
                                vrep.simx_opmode_blocking)
                    # Apply action
                    # Actuate tip target position
                    # if tip_tgt_rot_use_world_frame
                    if actions[i, 6] > 0:
                        _, gripper_tt_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_tt_handle, -1,
                                vrep.simx_opmode_blocking)
                        newPos = np.array(gripper_tt_pos) + actions[i, :3]
                        vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, -1, list(newPos),
                                vrep.simx_opmode_blocking)
                    else:
                        vrep.simxSetObjectPosition(self.client_ID, self.gripper_tt_handle, self.gripper_tt_handle,
                                list(actions[i, :3]), vrep.simx_opmode_blocking)
                    # Actuate tip target orientation
                    vrep.simxSetObjectOrientation(self.client_ID, self.gripper_tt_handle, self.gripper_tt_handle,
                            list(actions[i, 3:6]), vrep.simx_opmode_blocking)

                # Actuate gripper
                # -1 == open, 1 == close
                gripper_vel = (actions[i, 8]/np.abs(actions[i, 8])) * self._gripper_closing_vel

                vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f1_handle, gripper_vel,
                        vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.client_ID, self.gripper_f2_handle, gripper_vel,
                        vrep.simx_opmode_blocking)
            else:
                # Real step
                # Update gripper_tt pos and orients to current GRIPPER (NOT Gripper TT) pos and orient (for tt reset)
                #print("step0")
                _, self._gripper_tt_pos = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1,
                        vrep.simx_opmode_buffer)
                _, self._gripper_tt_orient = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1,
                        vrep.simx_opmode_buffer)
                #print("step1")
                vrep.simxSynchronousTrigger(self.client_ID)
                #print("step11")
                # make sure all commands are exeucted
                vrep.simxGetPingTime(self.client_ID)
                # obtain next state
                #print("step12")
                next_state = self.getCurrentState()
                #print("step13")
                next_states.append(next_state.reshape(1, -1))
                # NOTE: actions is not relevant in calculating rewards
                rewards.append(self.getRewards(self.state, actions[:, :7], next_state))
                #print("step14")
                #print("step2")
                self._step += 1
                self.state = np.copy(next_state)
                if self._isDone():
                    break

        if len(next_states) > 0:
            next_states = np.concatenate(next_states, axis=0)
        else:
            next_states = None
        if len(rewards) > 0:
            rewards = np.concatenate(rewards, axis=0)
        else:
            next_states = None
        return next_states, rewards, self._isDone(), None

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
    elif env_name == "VREPPushTask7DoF":
        return VREPPushTask7DoFEnvironment(
                *args,
                **kwargs,
                mico_model_path="models/robots/non-mobile/MicoRobot7DoF.ttm")
    elif env_name == "VREPPushTask7DoFIK":
        return VREPPushTask7DoFIKEnvironment(
                *args,
                **kwargs,
                mico_model_path="models/robots/non-mobile/MicoRobot7DoFIK.ttm")
    elif env_name == "VREPPushTask7DoFSparseRewards":
        return VREPPushTask7DoFSparseRewardsEnvironment(
                *args,
                **kwargs,
                )
    elif env_name == "VREPPushTask7DoFSparseRewardsIK":
        return VREPPushTask7DoFSparseRewardsIKEnvironment(
                *args,
                **kwargs,
                mico_model_path="models/robots/non-mobile/MicoRobot7DoFIK.ttm",
                )
    elif env_name == "VREPGraspTask7DoF":
        return VREPGraspTask7DoFEnvironment(
                *args,
                **kwargs,
                )
    elif env_name == "VREPGraspTask7DoFSparseRewards":
        return VREPGraspTask7DoFSparseRewardsEnvironment(
                *args,
                **kwargs,
                )
    elif env_name == "VREPGraspTask7DoFSparseRewardsIK":
        return VREPGraspTask7DoFSparseRewardsIKEnvironment(
                *args,
                **kwargs,
                )
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
