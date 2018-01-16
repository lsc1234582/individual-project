import argparse
import numpy as np
import random
import vrep

def generateRandomVel(max_vel):
    """
        Generate an array of shape (6,) of range [-max_vel, max_vel].
    """
    return np.array([random.random() * max_vel * 2 - max_vel for _ in range(6)])

def getCost(state):
    """
        Return the sum of the Euclidean distance between gripper and cuboid and the Euclidean distance between cuboid and targetPlane.
    """
    return np.sqrt(np.sum(np.square(state[-6:-3]))) + np.sqrt(np.sum(np.square(state[-3:])))

def getCurrentState(client_ID, joint_handles, gripper_handle, cuboid_handle, target_plane_handle):
    """
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

def standardise(arr, mean, std):
    return np.divide((arr - mean),  std, out=np.zeros_like(arr), where=std!=0)

def invStandardise(arr, mean, std):
    return arr * std + mean

def mse(arr1, arr2):
    return np.mean(np.square(arr1 - arr2))

def checkPositive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} is not a positive value".format(value))
    return ivalue

MAX_JOINT_VELOCITY = 1.0
INITIAL_JOINT_POSITIONS = [np.pi, 1.5 * np.pi, 1.5 * np.pi, np.pi, np.pi, np.pi]
INITIAL_CUBOID_POSITION = [0.3, 0.5, 0.05]
