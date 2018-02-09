import numpy as np
import vrep

class VREPPushTaskEnvironment():
    MAX_JOINT_VELOCITY_DELTA = 1.0
    MAX_JOINT_VELOCITY = 6.0
    #INITIAL_JOINT_POSITIONS = [np.pi, 1.5 * np.pi, 1.5 * np.pi, np.pi, np.pi, np.pi]
    #INITIAL_CUBOID_POSITION = [0., 0.5, 0.05]
    INITIAL_JOINT_POSITIONS = [np.pi, 1.5 * np.pi, 1.5 * np.pi, np.pi, np.pi, np.pi]
    INITIAL_CUBOID_POSITION = [0.3, 0.5, 0.05]

    def __init__(self):
        pass

    def __enter__(self):
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.client_ID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP

         # enable the synchronous mode on the client:
        vrep.simxSynchronous(self.client_ID,True)
        # start the simulation:
        vrep.simxStartSimulation(self.client_ID, vrep.simx_opmode_blocking)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # stop the simulation:
        vrep.simxStopSimulation(self.client_ID, vrep.simx_opmode_blocking)
        # before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(self.client_ID)
        # disconnect
        vrep.simxFinish(self.client_ID)
        return False

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


    def reset(self, is_first_reset):
        """
        Reset the environment
        Return initial state

        """

        if not is_first_reset:
            # tear down datastreams
            for i in range(6):
                _, _ = vrep.simxGetObjectFloatParameter(self.client_ID, self.joint_handles[i], 2012, vrep.simx_opmode_discontinue)
                _, _ = vrep.simxGetJointPosition(self.client_ID, self.joint_handles[i],
                        vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectPosition(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectOrientation(self.client_ID, self.gripper_handle, -1, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectPosition(self.client_ID, self.cuboid_handle, -1, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectPosition(self.client_ID, self.target_plane_handle, -1, vrep.simx_opmode_discontinue)

            # remove Mico
            vrep.simxRemoveModel(self.client_ID, self.model_base_handle, vrep.simx_opmode_blocking)

        # get handles
        _, self.cuboid_handle = vrep.simxGetObjectHandle(self.client_ID, 'Cuboid', vrep.simx_opmode_blocking)
        _, self.target_plane_handle = vrep.simxGetObjectHandle(self.client_ID, 'TargetPlane', vrep.simx_opmode_blocking)

        _, self.model_base_handle = vrep.simxLoadModel(self.client_ID, 'models/robots/non-mobile/MicoRobot.ttm', 0, vrep.simx_opmode_blocking)
        self.joint_handles = [-1, -1, -1, -1, -1, -1]
        for i in range(6):
            _, self.joint_handles[i] = vrep.simxGetObjectHandle(self.client_ID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
        _, self.gripper_handle = vrep.simxGetObjectHandle(self.client_ID, 'MicoHand', vrep.simx_opmode_blocking)

        # initialise mico joint positions, cuboid orientation and cuboid position
        vrep.simxPauseCommunication(self.client_ID, 1)
        for i in range(6):
            vrep.simxSetJointPosition(self.client_ID, self.joint_handles[i], VREPPushTaskEnvironment.INITIAL_JOINT_POSITIONS[i], vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.client_ID, self.cuboid_handle, -1, [0, 0, 0], vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.client_ID, self.cuboid_handle, -1, VREPPushTaskEnvironment.INITIAL_CUBOID_POSITION, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.client_ID, 0)
        vrep.simxGetPingTime(self.client_ID)

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

        # obtain first state
        current_state = self.getCurrentState(self.client_ID, self.joint_handles, self.gripper_handle, self.cuboid_handle,
                self.target_plane_handle)

        self.state = current_state

        return current_state

    def getRewards(self, state):
        """
            Return the sum of the Euclidean distance between gripper and cuboid and the Euclidean distance between cuboid and targetPlane.
        """
        return -np.sqrt(np.sum(np.square(state[-6:-3]))) + np.sqrt(np.sum(np.square(state[-3:])))



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
            for i in range(6):
                vrep.simxSetJointTargetVelocity(self.client_ID, self.joint_handles[i], current_vel[i],
                        vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.client_ID, 0)
            vrep.simxSynchronousTrigger(self.client_ID)
            vrep.simxSynchronousTrigger(self.client_ID)
            vrep.simxSynchronousTrigger(self.client_ID)
            vrep.simxSynchronousTrigger(self.client_ID)
            vrep.simxSynchronousTrigger(self.client_ID)
            # make sure all commands are exeucted
            vrep.simxGetPingTime(self.client_ID)
            # obtain next state
            next_state = self.getCurrentState(self.client_ID, self.joint_handles, self.gripper_handle, self.cuboid_handle,
                    self.target_plane_handle)
            next_states.append(next_state)
            rewards.append(self.getRewards(next_state))
            self.state = next_state[:]

        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        return next_states, rewards


