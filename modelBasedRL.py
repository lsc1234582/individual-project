#! /usr/bin/python3
import argparse
import logging
import numpy as np
import os
import random
import sys
import vrep
from keras.models import Sequential
from keras.models import load_model
from keras.layers import BatchNormalization, Dense
from keras.callbacks import TensorBoard
from common import *

T_INS_FILE = 'data/T_ins_100hz_500_200'
T_OUTS_FILE = 'data/T_outs_100hz_500_200'

SHORT_HOR_LENGTH = 5
SHORT_HOR_NUM = 20
GAMMA = 0.5

DATASET_SIZE = 20000 # Max number of data points to consider
VAL_SPLIT = 0.2
NEW_SPLIT = 0.5      # The proportion of the training data that comes from T_new
EPSILON = 0.1

BATCH_SIZE = 128
EPOCHS = 100

def get_model():
    model = Sequential([
            Dense(96, input_dim=30, kernel_initializer='normal', activation='relu'),
            BatchNormalization(),
            Dense(48, kernel_initializer='normal', activation='relu'),
            BatchNormalization(),
            Dense(24, kernel_initializer='normal')
        ])
    model.compile(loss='mse', optimizer='rmsprop')
    return model

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Model-based Reinforcement Learning on Mico Robot Arm.")
    arg_parser.add_argument("eps_length", type=checkPositive, help="The length of an episode; must be positive.")
    arg_parser.add_argument("num_eps", type=checkPositive, help="Number of episodes to run; must be positive.")
    arg_parser.add_argument("--monitor-log-file", "-m", dest="monitor_log_path", help="The path to the the\
            Tensorboard monitor log.")
    arg_parser.add_argument("--model-file", "-f", dest="model_file", required=True, help="The path to the file of the \
            learned model; if --new-model is specified this is the path the new model is saved to.")
    arg_parser.add_argument("--new-model", "-n", dest="new_model", action="store_true", help="A new model is learned \
            and saved instead of using an existing model.")
    arg_parser.add_argument("--overwrite-model-file", "-o", dest="overwrite_model_file", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--log-level", dest="log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = arg_parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log value:{}".format(log_level))
    logging.basicConfig(level=log_level)

    if args.monitor_log_path and not os.path.exists(args.monitor_log_path):
        logging.critical("Monitor log path {} does not exist.".format(args.monitor_log_path))
        logging.info("Program ended")
        sys.exit(1)
    if args.new_model and os.path.exists(args.model_file) and not args.overwrite_model_file:
        logging.critical("Model file {} already exists. Add flag --overwrite-model-file if want to overwrite the \
                file".format(args.model_file))
        logging.info("Program ended")
        sys.exit(1)
    elif not args.new_model and not os.path.exists(args.model_file):
        logging.critical("Model file {} does not exist.".format(args.model_file))
        logging.info("Program ended")
        sys.exit(1)

    random.seed(args.seed)

    logging.info("Program {} started".format(arg_parser.prog))
    vrep.simxFinish(-1) # just in case, close all opened connections
    client_ID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP

    if args.new_model:
        # load random trajectories
        T_ins = np.load(T_INS_FILE + ".npy")
        T_outs = np.load(T_OUTS_FILE + ".npy")
        NUM_SAMPLES = min(T_ins.shape[0], DATASET_SIZE)
        logging.info("Dataset of size {} loaded, only {} of the total are retained".format(T_ins.shape[0],
            NUM_SAMPLES))
        NUM_VAL = int(NUM_SAMPLES * VAL_SPLIT)
        NUM_TRAIN = NUM_SAMPLES - NUM_VAL
        np.random.seed(args.seed)
        np.random.shuffle(T_ins)
        np.random.seed(args.seed)
        np.random.shuffle(T_outs)
        T_ins_train = T_ins[:NUM_TRAIN]
        T_outs_train = T_outs[:NUM_TRAIN]
        T_ins_train_mean = np.mean(T_ins_train, 0)
        T_ins_train_std = np.std(T_ins_train, 0)
        T_outs_train_mean = np.mean(T_outs_train, 0)
        T_outs_train_std = np.std(T_outs_train, 0)
        # save training set stats
        np.save(T_INS_FILE + "_train_mean.npy", T_ins_train_mean)
        np.save(T_INS_FILE + "_train_std.npy", T_ins_train_std)
        np.save(T_OUTS_FILE + "_train_mean.npy", T_outs_train_mean)
        np.save(T_OUTS_FILE + "_train_std.npy", T_outs_train_std)

        T_ins_train_norm = standardise(T_ins_train, T_ins_train_mean, T_ins_train_std)
        T_outs_train_norm = standardise(T_outs_train, T_outs_train_mean, T_outs_train_std)

        T_ins_val = T_ins[NUM_TRAIN: NUM_TRAIN + NUM_VAL]
        T_outs_val = T_outs[NUM_TRAIN: NUM_TRAIN + NUM_VAL]
        T_ins_val_norm = standardise(T_ins_val, T_ins_train_mean, T_ins_train_std)
        T_outs_val_norm = standardise(T_outs_val, T_outs_train_mean, T_outs_train_std)
        logging.info("Random trajectories loaded")
        # construct and compile T_model
        T_model = get_model()
        # pretrain model on old trajectories
        # add zero-mean gaussian noise to training data to make the model more robust
        X = T_ins_train_norm + np.random.normal(0, 0.05, T_ins_train_norm.shape)
        y = T_outs_train_norm + np.random.normal(0, 0.05, T_outs_train_norm.shape)
        if args.monitor_log_path:
            # monitor training
            monitor_log_iter_path = args.monitor_log_path + "/PRE"
            if not os.path.exists(monitor_log_iter_path):
                os.mkdir(monitor_log_iter_path)
            tf_board_monitor = TensorBoard(log_dir=monitor_log_iter_path, histogram_freq=0, batch_size=BATCH_SIZE,
                    write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
            # train T_model
            T_model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(T_ins_val_norm, T_outs_val_norm), callbacks=[tf_board_monitor])
        else:
            # train T_model
            T_model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(T_ins_val_norm, T_outs_val_norm))
    else:
        # load training set stats
        T_ins_train_mean = np.load(T_INS_FILE + "_train_mean.npy")
        T_ins_train_std = np.load(T_INS_FILE + "_train_std.npy")
        T_outs_train_mean = np.load(T_OUTS_FILE + "_train_mean.npy")
        T_outs_train_std = np.load(T_OUTS_FILE + "_train_std.npy")
        # load model
        T_model = load_model(args.model_file)

    if client_ID!=-1:
        logging.info('Connected to remote API server')

        # enable the synchronous mode on the client:
        vrep.simxSynchronous(client_ID,True)

        # start the simulation:
        vrep.simxStartSimulation(client_ID, vrep.simx_opmode_blocking)

        _, cuboid_handle = vrep.simxGetObjectHandle(client_ID, 'Cuboid', vrep.simx_opmode_blocking)
        _, target_plane_handle = vrep.simxGetObjectHandle(client_ID, 'TargetPlane', vrep.simx_opmode_blocking)

        # init new trajectories
        T_new_ins_list = []
        T_new_outs_list = []

        for i in range(args.num_eps):
            logging.info("{}th iteration".format(i))
            if args.monitor_log_path:
                monitor_log_iter_path = args.monitor_log_path + "/ITER_{}".format(i)
                if not os.path.exists(monitor_log_iter_path):
                    os.mkdir(monitor_log_iter_path)
            # get handles
            _, model_base_handle = vrep.simxLoadModel(client_ID, 'models/robots/non-mobile/MicoRobot.ttm', 0, vrep.simx_opmode_blocking)
            joint_handles = [-1, -1, -1, -1, -1, -1]
            for i in range(6):
                _, joint_handles[i] = vrep.simxGetObjectHandle(client_ID, 'Mico_joint' + str(i+1),vrep.simx_opmode_blocking)
            _, gripper_handle = vrep.simxGetObjectHandle(client_ID, 'MicoHand', vrep.simx_opmode_blocking)

            # initialise mico joint positions, cuboid orientation and cuboid position
            vrep.simxPauseCommunication(client_ID, 1)
            for i in range(6):
                vrep.simxSetJointPosition(client_ID, joint_handles[i], INITIAL_JOINT_POSITIONS[i], vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(client_ID, cuboid_handle, -1, [0, 0, 0], vrep.simx_opmode_oneshot)
            vrep.simxSetObjectPosition(client_ID, cuboid_handle, -1, INITIAL_CUBOID_POSITION, vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(client_ID, 0)
            vrep.simxGetPingTime(client_ID)

            current_vel = np.array([0, 0, 0, 0, 0, 0], dtype='float')
            joint_angles = np.array([0, 0, 0, 0, 0, 0], dtype='float')

            # set up datastreams
            for i in range(6):
                _, current_vel[i] = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012,
                        vrep.simx_opmode_streaming)
                _, joint_angles[i] = vrep.simxGetJointPosition(client_ID, joint_handles[i], vrep.simx_opmode_streaming)
            _, gripper_pos = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_streaming)
            _, gripper_orient = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_streaming)
            _, cuboid_pos = vrep.simxGetObjectPosition(client_ID, cuboid_handle, -1, vrep.simx_opmode_streaming)
            _, target_plane_pos = vrep.simxGetObjectPosition(client_ID, target_plane_handle, -1, vrep.simx_opmode_streaming)

            # destroy dummy arrays for setting up the datastream
            del current_vel, joint_angles, gripper_pos, gripper_orient, cuboid_pos, target_plane_pos

            # obtain first state
            current_state = getCurrentState(client_ID, joint_handles, gripper_handle, cuboid_handle,
                    target_plane_handle)

            for step in range(args.eps_length):
                # select best action through random sampling
                min_trajectory_cost = float('inf')
                opt_action = generateRandomVel(MAX_JOINT_VELOCITY)
                for _ in range(SHORT_HOR_NUM):
                    action_trace = []
                    trajectory_cost = 0
                    state = np.copy(current_state)
                    for short_step in range(SHORT_HOR_LENGTH):
                        action = generateRandomVel(MAX_JOINT_VELOCITY)
                        action_trace.append(action)
                        T_in = np.concatenate([state, action])
                        T_in = standardise(T_in, T_ins_train_mean, T_ins_train_std)
                        T_out = T_model.predict(np.reshape(T_in, (1, -1)))
                        T_out = invStandardise(T_out, T_outs_train_mean, T_outs_train_std)
                        state += T_out.flatten()
                        cost = getCost(state)
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
                current_vel = current_state[:6] + opt_action
                vrep.simxPauseCommunication(client_ID, 1);
                for i in range(6):
                    vrep.simxSetJointTargetVelocity(client_ID, joint_handles[i], current_vel[i],
                            vrep.simx_opmode_oneshot)
                vrep.simxPauseCommunication(client_ID, 0);
                vrep.simxSynchronousTrigger(client_ID);
                vrep.simxSynchronousTrigger(client_ID);
                # make sure all commands are exeucted
                vrep.simxGetPingTime(client_ID)
                # obtain next state
                next_state = getCurrentState(client_ID, joint_handles, gripper_handle, cuboid_handle,
                        target_plane_handle)

                if args.new_model:
                    # enrich trajectory dataset
                    T_in = np.concatenate([current_state, opt_action])
                    T_out = next_state - current_state
                    T_new_ins_list.append(T_in[np.newaxis, :])
                    T_new_outs_list.append(T_out[np.newaxis, :])

                # proceed to next state
                current_state = next_state[:]

            if args.new_model:
                # combine new and old trajectories
                T_new_ins = np.concatenate(T_new_ins_list)
                T_new_outs = np.concatenate(T_new_outs_list)
                T_new_ins_mean = np.mean(T_new_ins, 0)
                T_new_ins_std = np.std(T_new_ins, 0)
                T_new_outs_mean = np.mean(T_new_outs, 0)
                T_new_outs_std = np.std(T_new_outs, 0)
                ins_mean_mse = mse(T_ins_train_mean, T_new_ins_mean)
                ins_std_mse = mse(T_ins_train_std, T_new_ins_std)
                outs_mean_mse = mse(T_outs_train_mean, T_new_outs_mean)
                outs_std_mse = mse(T_outs_train_std, T_new_outs_std)
                if ins_mean_mse >= EPSILON or ins_std_mse >= EPSILON or outs_mean_mse >= EPSILON or outs_std_mse >= EPSILON:
                    logging.warn("Significant training distribution shift")
                    logging.debug("ins_mean_mse: {:.3} ins_std_mse: {:.3} outs_mean_mse: {:.3} outs_std_mse:\
                            {:.3}".format(ins_mean_mse, ins_std_mse, outs_mean_mse, outs_std_mse))

                T_new_ins = standardise(T_new_ins, T_ins_train_mean, T_ins_train_std)
                T_new_outs = standardise(T_new_outs, T_outs_train_mean, T_outs_train_std)
                NUM_TRAIN_NEW = min(int(NUM_TRAIN * NEW_SPLIT), T_new_ins.shape[0])
                NUM_TRAIN_OLD = NUM_TRAIN - NUM_TRAIN_NEW

                np.random.seed(args.seed)
                np.random.shuffle(T_new_ins)
                np.random.seed(args.seed)
                np.random.shuffle(T_new_outs)
                np.random.seed(args.seed)
                np.random.shuffle(T_ins_train_norm)
                np.random.seed(args.seed)
                np.random.shuffle(T_outs_train_norm)

                X = np.concatenate([T_new_ins[:NUM_TRAIN_NEW], T_ins_train_norm[:NUM_TRAIN_OLD]])
                y = np.concatenate([T_new_outs[:NUM_TRAIN_NEW], T_outs_train_norm[:NUM_TRAIN_OLD]])

                # add zero-mean gaussian noise
                X += np.random.normal(0, 0.05, X.shape)
                y += np.random.normal(0, 0.05, y.shape)

                if args.monitor_log_path:
                    # monitor training
                    tf_board_monitor = TensorBoard(log_dir=monitor_log_iter_path, histogram_freq=0,
                            batch_size=BATCH_SIZE,
                            write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
                    # train T_model
                    T_model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(T_ins_val_norm, T_outs_val_norm), callbacks=[tf_board_monitor])
                else:
                    # train T_model
                    T_model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(T_ins_val_norm, T_outs_val_norm))

            # tear down datastreams
            for i in range(6):
                _, _ = vrep.simxGetObjectFloatParameter(client_ID, joint_handles[i], 2012, vrep.simx_opmode_discontinue)
                _, _ = vrep.simxGetJointPosition(client_ID, joint_handles[i],
                        vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectPosition(client_ID, gripper_handle, -1, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectOrientation(client_ID, gripper_handle, -1, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectPosition(client_ID, cuboid_handle, -1, vrep.simx_opmode_discontinue)
            _, _ = vrep.simxGetObjectPosition(client_ID, target_plane_handle, -1, vrep.simx_opmode_discontinue)

            # remove Mico
            vrep.simxRemoveModel(client_ID, model_base_handle, vrep.simx_opmode_blocking)

        # stop the simulation:
        vrep.simxStopSimulation(client_ID, vrep.simx_opmode_blocking)
        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(client_ID)

        # Now close the connection to V-REP:
        vrep.simxFinish(client_ID)
        if args.new_model:
            T_model.save(args.model_file)
    else:
        logging.critical("Failed connecting to remote API server")

    logging.info('Program ended')
