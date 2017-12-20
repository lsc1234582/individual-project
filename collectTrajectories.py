#! /usr/bin/python3
# Copyright 2006-2017 Coppelia Robotics GmbH. All rights reserved.
# marc@coppeliarobotics.com
# www.coppeliarobotics.com
#
# -------------------------------------------------------------------
# THIS FILE IS DISTRIBUTED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
# WARRANTY. THE USER WILL USE IT AT HIS/HER OWN RISK. THE ORIGINAL
# AUTHORS AND COPPELIA ROBOTICS GMBH WILL NOT BE LIABLE FOR DATA LOSS,
# DAMAGES, LOSS OF PROFITS OR ANY OTHER KIND OF LOSS WHILE USING OR
# MISUSING THIS SOFTWARE.
#
# You are free to use/modify/distribute this file for whatever purpose!
# -------------------------------------------------------------------
#
# This file was automatically created for V-REP release V3.4.0 rev. 1 on April 5th 2017

# Make sure to have the server side running in V-REP:
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simExtRemoteApiStart(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!
import random
import pprint
import numpy as np
import sys
import time

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')


def generateRandomVel(max_vel):
    return np.array([random.random() * max_vel * 2 - max_vel for _ in range(6)])

def resetMicoState(client_ID, joint_handles):
    for i in range(6):
        vrep.simxSetJointTargetPosition(client_ID, joint_handles[i], np.pi, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(client_ID);
    time.sleep(1.0)
    for i in range(6):
        vrep.simxSetJointTargetVelocity(client_ID, joint_handles[i], 0, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(client_ID);
    print('Mico state reset')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Format: collectTrajectories.py episode_length num_episodes [seed]', file=sys.stderr)
        sys.exit(-1)
    eps_length = int(sys.argv[1])
    num_eps = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) == 4 else time.time()
    random.seed(seed)

    print ('Program started')
    vrep.simxFinish(-1) # just in case, close all opened connections
    client_ID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
    # Dynamics model input and output arrays
    T_ins = []
    T_outs = []
    if client_ID!=-1:
        print ('Connected to remote API server')

        # enable the synchronous mode on the client:
        vrep.simxSynchronous(client_ID,True)

        _, model_base_handle = vrep.simxLoadModel(client_ID, 'models/robots/non-mobile/MicoRobot.ttm', 0, vrep.simx_opmode_blocking)

        # start the simulation:
        vrep.simxStartSimulation(client_ID, vrep.simx_opmode_blocking)

        for i in range(num_eps):
            print("%d th iteration" % (i))
            joint_handles = [-1, -1, -1, -1, -1, -1]
            for i in range(6):
                _, joint_handles[i] = vrep.simxGetObjectHandle(client_ID, 'Mico_joint' + str(i+1),vrep.simx_opmode_blocking)

            _, gripper_handle = vrep.simxGetObjectHandle(client_ID, 'MicoHand', vrep.simx_opmode_blocking)
            current_vel = np.array([0, 0, 0, 0, 0, 0], dtype='float')
            joint_angles = np.array([0, 0, 0, 0, 0, 0], dtype='float')

            # obtain first state
            for i in range(6):
                _, current_vel[i] = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012, vrep.simx_opmode_blocking)
                _, joint_angles[i] = vrep.simxGetJointPosition(client_ID, joint_handles[i], vrep.simx_opmode_blocking)
            _, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
            _, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
            gripper_pos = np.array(gripper_pos)
            gripper_orient = np.array(gripper_orient)

            current_state = np.concatenate([current_vel, joint_angles, gripper_pos, gripper_orient])

            for step in range(eps_length):
                action = generateRandomVel(2)
                T_in = np.concatenate([current_state, action])

                current_vel += action
                vrep.simxPauseCommunication(client_ID, 1);
                for i in range(6):
                    vrep.simxSetJointTargetVelocity(client_ID, joint_handles[i], current_vel[i],
                            vrep.simx_opmode_oneshot)
                vrep.simxPauseCommunication(client_ID, 0);
                vrep.simxSynchronousTrigger(client_ID);
                # make sure all commands are exeucted
                vrep.simxGetPingTime(client_ID)
                # obtain next state
                for i in range(6):
                    _, current_vel[i] = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012, vrep.simx_opmode_blocking)
                    _, joint_angles[i] = vrep.simxGetJointPosition(client_ID, joint_handles[i], vrep.simx_opmode_blocking)
                _, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
                _, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
                gripper_pos = np.array(gripper_pos)
                gripper_orient = np.array(gripper_orient)

                next_state = np.concatenate([current_vel, joint_angles, gripper_pos, gripper_orient])
                T_out = next_state - current_state
                T_ins.append(T_in[np.newaxis, :])
                T_outs.append(T_out[np.newaxis, :])
                #print("T_in: ")
                #pprint.pprint(T_in)
                #print("T_out: ")
                #pprint.pprint(T_out)
                print("gripper_pos")
                pprint.pprint(gripper_pos)

                # proceed to next state
                current_state = next_state[:]
                time.sleep(0.5)

            # Reset Mico State
            vrep.simxRemoveModel(client_ID, model_base_handle, vrep.simx_opmode_blocking)
            _, model_base_handle = vrep.simxLoadModel(client_ID, 'models/robots/non-mobile/MicoRobot.ttm', 0, vrep.simx_opmode_blocking)

        # stop the simulation:
        vrep.simxStopSimulation(client_ID, vrep.simx_opmode_blocking)
        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(client_ID)

        # Now close the connection to V-REP:
        vrep.simxFinish(client_ID)
    else:
        print ('Failed connecting to remote API server')
    T_ins = np.concatenate(T_ins)
    T_outs = np.concatenate(T_outs)
    np.save('data/T_ins', T_ins)
    np.save('data/T_outs', T_outs)
    print('Trajectories saved')
    print ('Program ended')
