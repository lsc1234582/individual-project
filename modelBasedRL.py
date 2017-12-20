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
import numpy as np
import sys
import time
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation

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

def getCost(state, target_pos):
    """
        Euclidean distance between gripper position and target position.
    """
    gripper_pos = state[-6: -3]
    #print('target_pos')
    #print(target_pos)
    #print('gripper_pos')
    #print(gripper_pos)
    return np.sqrt(np.sum(np.square(gripper_pos - target_pos)))

def resetMicoState(client_ID, joint_handles):
    for i in range(6):
        vrep.simxSetJointTargetPosition(client_ID, joint_handles[i], np.pi, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(client_ID);
    time.sleep(1.0)
    for i in range(6):
        vrep.simxSetJointTargetVelocity(client_ID, joint_handles[i], 0, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(client_ID);
    print('Mico state reset')


SHORT_HOR_LENGTH = 5
SHORT_HOR_NUM = 200
GAMMA = 0.5

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Format: modelBasedRL.py episode_length num_episodes new_model(0|1) [seed]', file=sys.stderr)
        sys.exit(-1)
    eps_length = int(sys.argv[1])
    num_eps = int(sys.argv[2])
    new_model = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) == 5 else time.time()
    random.seed(seed)

    print ('Program started')
    vrep.simxFinish(-1) # just in case, close all opened connections
    client_ID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
    # load random trajectories
    T_ins = np.load('data/T_ins.npy')
    T_outs = np.load('data/T_outs.npy')

    if new_model == 1:
        # construct and compile T_model
        T_model = Sequential([
            Dense(32, input_dim=18, kernel_initializer='normal'),
            Activation('relu'),
            Dense(12, kernel_initializer='normal'),
            Activation('linear')
            ])
        T_model.compile(optimizer='sgd', loss='mse')
    else:
        # load model
        T_model = load_model('model/T_model.h5')

    if client_ID!=-1:
        print ('Connected to remote API server')

        # enable the synchronous mode on the client:
        vrep.simxSynchronous(client_ID,True)

        _, model_base_handle = vrep.simxLoadModel(client_ID, 'models/robots/non-mobile/MicoRobot.ttm', 0, vrep.simx_opmode_blocking)

        # start the simulation:
        vrep.simxStartSimulation(client_ID, vrep.simx_opmode_blocking)

        for i in range(num_eps):
            print("%d th iteration" % (i))
            # get handles
            joint_handles = [-1, -1, -1, -1, -1, -1]
            for i in range(6):
                _, joint_handles[i] = vrep.simxGetObjectHandle(client_ID, 'Mico_joint' + str(i+1),vrep.simx_opmode_blocking)

            _, gripper_handle = vrep.simxGetObjectHandle(client_ID, 'MicoHand', vrep.simx_opmode_blocking)
            _, target_handle = vrep.simxGetObjectHandle(client_ID, 'Target', vrep.simx_opmode_blocking)
            current_vel = np.array([0, 0, 0, 0, 0, 0], dtype='float')
            if new_model == 1:
                # train T_model
                T_model.fit(T_ins, T_outs, batch_size=32, epochs=10)
            # obtain first state
            for i in range(6):
                _, current_vel[i] = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012, vrep.simx_opmode_blocking)
            _, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
            _, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
            _, target_pos = vrep.simxGetObjectPosition(client_ID, target_handle, -1, vrep.simx_opmode_blocking)
            gripper_pos = np.array(gripper_pos)
            gripper_orient = np.array(gripper_orient)
            target_pos = np.array(target_pos)
            current_state = np.concatenate([current_vel, gripper_pos, gripper_orient])
            for step in range(eps_length):
                # select best action through random sampling
                min_trajectory_cost = float('inf')
                opt_action = generateRandomVel(1)
                for _ in range(SHORT_HOR_NUM):
                    action_trace = []
                    trajectory_cost = 0
                    state = current_state[:]
                    for short_step in range(SHORT_HOR_LENGTH):
                        action = generateRandomVel(2)
                        action_trace.append(action)
                        T_in = np.concatenate([state, action])
                        T_out = T_model.predict(np.reshape(T_in, (1, -1)), batch_size=1)
                        state += T_out.flatten()
                        cost = getCost(state, target_pos)
                        trajectory_cost += cost * (GAMMA ** short_step)
                        #print('T_in')
                        #print(T_in)
                        #print('T_out')
                        #print(T_out)
                        #print('cost %f' % (cost))
                    if trajectory_cost < min_trajectory_cost:
                        min_trajectory_cost = trajectory_cost
                        #print('min_trajectory_cost %f' % (min_trajectory_cost))
                        opt_action = action_trace[0]

                # execute one-step optimal action
                current_vel += opt_action
                for i in range(6):
                    vrep.simxSetJointTargetVelocity(client_ID, joint_handles[i], current_vel[i],
                            vrep.simx_opmode_blocking)
                vrep.simxSynchronousTrigger(client_ID);

                # obtain next state
                for i in range(6):
                    _, current_vel[i] = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012, vrep.simx_opmode_blocking)
                _, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
                _, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_blocking)
                gripper_pos = np.array(gripper_pos)
                gripper_orient = np.array(gripper_orient)
                next_state = np.concatenate([current_vel, gripper_pos, gripper_orient])

                # enrich trajectory dataset
                T_in = np.concatenate([current_state, opt_action])
                T_out = next_state - current_state
                T_ins = np.concatenate([T_ins, T_in[np.newaxis, :]])
                T_outs = np.concatenate([T_outs, T_out[np.newaxis, :]])

                # proceed to next state
                current_state = next_state[:]

            # Reset Mico State
            vrep.simxRemoveModel(client_ID, model_base_handle, vrep.simx_opmode_blocking)
            _, model_base_handle = vrep.simxLoadModel(client_ID, 'models/robots/non-mobile/MicoRobot.ttm', 0, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(client_ID);

        # stop the simulation:
        vrep.simxStopSimulation(client_ID, vrep.simx_opmode_blocking)
        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(client_ID)

        # Now close the connection to V-REP:
        vrep.simxFinish(client_ID)
    else:
        print ('Failed connecting to remote API server')

    if new_model == 1:
        T_model.save('model/T_model.h5')
    print ('Program ended')
