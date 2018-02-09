"""
Python script to train a Mico pushing task controller from scratch
"""
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import vrep
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import BatchNormalization, Dense, Input
from keras.callbacks import TensorBoard
from keras import backend as K
from common import *
from GaussianPolicy import GaussianPolicy
from VREPEnvironments import VREPPushTaskEnvironment
import json
import time

def getModel():
    """
    Model for transition dynamics, rewards and termination of episodes.
    Inputs: [PrevState(24,), Action(6,)](30,)
    Outputs: [NestState-PrevState(24,), Reward](25,)
    """
    prevState_action_l = Input(shape=(30,), dtype="float32", name="prevState_action_l")
    H_l = Dense(256, kernel_initializer="normal", activation="relu", name="hidden_l1")(prevState_action_l)
    H_l = BatchNormalization()(H_l)
    H_l = Dense(64, kernel_initializer="normal", activation="relu", name="hidden_l2")(H_l)
    H_l = BatchNormalization()(H_l)
    nextState_reward_l = Dense(25, kernel_initializer="normal", name="nextState_l")(H_l)
    #dest_l = Dense(1, kernel_initializer="normal", activation="sigmoid", name="dest_l")(H_l)
    model = Model(inputs=prevState_action_l, outputs=nextState_reward_l)
    model.compile(loss="mse", optimizer="rmsprop")
    return model

def getPolicy():
    return GaussianPolicy()


def getAdvantages(rewards, discount_factor):
    eps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running_discounted_advantages = 0
    for i in range(eps - 1, -1, -1):
        running_discounted_advantages = running_discounted_advantages * discount_factor + rewards[i]
        advantages[i] = running_discounted_advantages
    return advantages


def getGreedyFactor(eps, max_vel, growth_rate, mid_point):
    """
    Greedy factor scheduler
    """
    return max_vel / (1 + np.exp(-growth_rate * (eps - mid_point)))

if __name__ == "__main__":
    SESSION_ROOT = "TrainingSessions"
    # Log reward sums, add entries to Summary
    ADDITIONAL_LOGGING = True
    NEW_MODEL = True
    SAMPLE_FROM_MODEL = False
    NEW_POLICY = True
    #SESSION_IDS = ["03", "04", "05", "06", "07", "08", "09", "10"]
    SESSION_IDS = ["11"]

    TRAINING_SUMMARY_FILE = "Summary.csv"
    # Write summary header
    if ADDITIONAL_LOGGING:
        with open(SESSION_ROOT + '/' + TRAINING_SUMMARY_FILE, 'a') as f:
            f.write("SessionID, NumTrainingSteps, TrainingWallClockTime, AverageLast20RewardSum\n")
    for SESSION_ID in SESSION_IDS:
        TRAINING_SESSION_ROOT = SESSION_ROOT + "/" + SESSION_ID + "/"
        MODEL_FILE = TRAINING_SESSION_ROOT + "Model.h5"
        POLICY_FILE = TRAINING_SESSION_ROOT + "Policy.h5"

        MODEL_TRAINING_LOG_FILE = TRAINING_SESSION_ROOT + "ModelTraining.log"
        POLICY_TRAINING_LOG_FILE = TRAINING_SESSION_ROOT + "PolicyTraining.log"
        REWARDS_SUM_LOG_FILE = TRAINING_SESSION_ROOT + "RewardSums.log"


        HYPER_PARAM_CONFIG_FILE = TRAINING_SESSION_ROOT + "HyperParamConfig.json"
        # TODO: if hyper param config file doesn't exist continue
        with open(HYPER_PARAM_CONFIG_FILE, 'r') as f:
            HYPER_PARAMS = json.load(f)

        EPS_LENGTH = HYPER_PARAMS["EPS_LENGTH"]
        NUM_EPS = HYPER_PARAMS["NUM_EPS"]
        # Random action seed for this whole training session.
        RANDOM_ACTION_SEED = HYPER_PARAMS["RANDOM_ACTION_SEED"]

        DISCOUNT_FACTOR = HYPER_PARAMS["DISCOUNT_FACTOR"]
        GREEDY_FACTOR_MAX_VEL = HYPER_PARAMS["GREEDY_FACTOR_MAX_VEL"]
        GREEDY_FACTOR_GROWTH_RATE = HYPER_PARAMS["GREEDY_FACTOR_GROWTH_RATE"]
        GREEDY_FACTOR_MID_POINT = HYPER_PARAMS["GREEDY_FACTOR_MID_POINT"]

        # Episode number after which the trajectory sampling alternates between trained model and VREP simulation
        SWITCH_POINT_EP = HYPER_PARAMS["SWITCH_POINT_EP"]

        MODEL_BATCH_SIZE = HYPER_PARAMS["MODEL_BATCH_SIZE"]
        MODEL_EPOCHS = HYPER_PARAMS["MODEL_EPOCHS"]

        POLICY_BATCH_SIZE = HYPER_PARAMS["POLICY_BATCH_SIZE"]
        POLICY_EPOCHS = HYPER_PARAMS["POLICY_EPOCHS"]

        MAX_MODEL_BUFFER_SIZE = HYPER_PARAMS["MAX_MODEL_BUFFER_SIZE"]
        MAX_POLICY_BUFFER_SIZE = HYPER_PARAMS["MAX_POLICY_BUFFER_SIZE"]

        random.seed(RANDOM_ACTION_SEED)

        # TRAINING SUMMARY STATS
        reward_sums = []
        num_training_steps = 0
        with VREPPushTaskEnvironment() as env:
            model = getModel() if NEW_MODEL else load_model(MODEL_FILE)
            policy = GaussianPolicy(model_file=POLICY_FILE, epochs=POLICY_EPOCHS, batch_size=POLICY_BATCH_SIZE) if NEW_POLICY\
                    else GaussianPolicy(model_file=POLICY_FILE, epochs=POLICY_EPOCHS, batch_size=POLICY_BATCH_SIZE, is_load_model=True)

            # obtain first state
            current_state = env.reset(True)[np.newaxis, :]

            # Initialise training data buffers and training data stats for model training
            if NEW_MODEL:
                model_Xs = []
                model_ys = []
                model_buff_size = 0
                model_Xs_mean = None
                model_Xs_std = None
                model_ys_mean = None
                model_ys_std = None
            else:
                model_Xs_mean = np.load(TRAINING_SESSION_ROOT + "Model_Xs_mean.npy")
                model_Xs_std = np.load(TRAINING_SESSION_ROOT + "Model_Xs_std.npy")
                model_ys_mean = np.load(TRAINING_SESSION_ROOT + "Model_ys_mean.npy")
                model_ys_std = np.load(TRAINING_SESSION_ROOT + "Model_ys_std.npy")

            if NEW_POLICY:
                policy_Xs = []
                policy_ys = []
                advantages = []
                policy_buff_size = 0
            #    policy_Xs_mean = None
            #    policy_Xs_std = None
            #    policy_ys_mean = None
            #     policy_ys_std = None
            #    policy_rs_mean = None
            #    policy_rs_std = None

            training_start_time = time.time()
            for i in range(NUM_EPS):
                print("{}th episode".format(i))
                # initialise trajectory
                states = [current_state]
                actions = []
                rewards = []
                greedy_factor = getGreedyFactor(i, GREEDY_FACTOR_MAX_VEL, GREEDY_FACTOR_GROWTH_RATE,
                        GREEDY_FACTOR_MID_POINT)
                print("Greedy Factor: {}".format(greedy_factor))
                # collect trajectory
                print(current_state.shape)
                for step in range(EPS_LENGTH):
                    if (not NEW_POLICY) or (np.random.rand() < greedy_factor):
                        #action = policy.sampleAction(invStandardise(current_state[np.newaxis, :], policy_Xs_mean, policy_Xs_std))
                        #action = invStandardise(action, policy_ys_mean, policy_ys_std)
                        action = policy.sampleAction(current_state)[np.newaxis, :]
                    else:
                        action = generateRandomVel(env.MAX_JOINT_VELOCITY_DELTA)[np.newaxis, :]
                    if SAMPLE_FROM_MODEL and model_Xs_mean is not None and model_Xs_std is not None and \
                                             model_ys_mean is not None and model_ys_std is not None:
                        #print(current_state.shape)
                        #print(action.shape)
                        X = np.concatenate([current_state, action], axis=1)
                        pred_next_state_reward = model.predict(standardise(X, model_Xs_mean, model_Xs_std))
                        pred_next_state_reward = invStandardise(pred_next_state_reward, model_ys_mean, model_ys_std)
                        next_state = pred_next_state_reward[:, :-1]
                        reward = pred_next_state_reward[:, -1]
                    else:
                        next_state, reward = env.step(action)
                    action = action.reshape(1, -1)
                    next_state = next_state.reshape(1, -1)
                    reward = reward.reshape(1, -1)
                    # Terminate if the current velocity is too high (avoid bad data)
                    if np.any(np.abs(next_state[:, :6]) >= env.MAX_JOINT_VELOCITY ):
                        break
                    actions.append(action)
                    states.append(next_state)
                    rewards.append(reward)
                    # proceed to next state
                    current_state = next_state

                    num_training_steps += 1

                # Move on to next episode if no good action is taken
                if len(actions) == 0:
                    continue
                states = np.concatenate(states, axis=0)
                actions = np.concatenate(actions, axis=0)
                rewards = np.concatenate(rewards, axis=0)

                # Record collected reward
                if ADDITIONAL_LOGGING:
                    with open(REWARDS_SUM_LOG_FILE, 'a') as f:
                        f.write("{}, {}\n".format(i, np.sum(rewards)))
                if NUM_EPS - i <= 20:
                    reward_sums.append(np.sum(rewards))

                if NEW_MODEL:
                    # X = [current_states(,24), actions(,6)]
                    # y = [next_states - current_states(,24), rewards]
                    X = np.concatenate([states[:-1, :], actions], axis=1)
                    y = np.concatenate([states[1:, :] - states[:-1, :], rewards], axis=1)
                    model_Xs.append(X)
                    model_ys.append(y)
                    model_buff_size += X.shape[0]

                    # Train model once the training data buffer is full
                    if model_buff_size >= MAX_MODEL_BUFFER_SIZE:
                        model_X = np.concatenate(model_Xs, axis=0)
                        model_y = np.concatenate(model_ys, axis=0)
                        # standardise training data
                        if model_Xs_mean is None or model_Xs_std is None or model_ys_mean is None or model_ys_std is None:
                            model_Xs_mean = np.mean(model_X, axis=0)
                            model_Xs_std = np.std(model_X, axis=0)
                            model_ys_mean = np.mean(model_y, axis=0)
                            model_ys_std = np.std(model_y, axis=0)
                            np.save(TRAINING_SESSION_ROOT + "Model_Xs_mean.npy", model_Xs_mean)
                            np.save(TRAINING_SESSION_ROOT + "Model_Xs_std.npy", model_Xs_std)
                            np.save(TRAINING_SESSION_ROOT + "Model_ys_mean.npy", model_ys_mean)
                            np.save(TRAINING_SESSION_ROOT + "Model_ys_std.npy", model_ys_std)
                        model_X = standardise(model_X, model_Xs_mean, model_Xs_std)
                        model_y = standardise(model_y, model_ys_mean, model_ys_std)
                        # add zero-mean gaussian noise
                        model_X += np.random.normal(0, 0.05, model_X.shape)
                        model_y += np.random.normal(0, 0.05, model_y.shape)
                        # train model
                        model.fit(model_X, model_y, batch_size=MODEL_BATCH_SIZE, epochs=MODEL_EPOCHS, validation_split=0.2,
                                 callbacks=[keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=10)
                                           ,keras.callbacks.CSVLogger(MODEL_TRAINING_LOG_FILE, append=True)
                                           ,keras.callbacks.TerminateOnNaN()])
                        # Empty model training data buffer
                        model_Xs = []
                        model_ys = []
                        model_buff_size = 0

                if NEW_POLICY:
                    policy_Xs.append(states[:-1, :])
                    policy_ys.append(actions[:])
                    advantages.append(getAdvantages(rewards, DISCOUNT_FACTOR))
                    policy_buff_size += actions.shape[0]
                    # Train policy once the training data buffer is full
                    if policy_buff_size >= MAX_POLICY_BUFFER_SIZE:
                        policy_X = np.concatenate(policy_Xs, axis=0)
                        policy_y = np.concatenate(policy_ys, axis=0)
                        advant = np.concatenate(advantages, axis=0)
                        # standardise advantage
                        advant = standardise(advant, np.mean(advant, axis=0), np.std(advant, axis=0))
                        """
                        if policy_Xs_mean is None:
                            policy_Xs_mean = np.mean(policy_X, 0)
                            policy_Xs_std = np.std(policy_X, 0)
                            policy_ys_mean = np.mean(policy_y, 0)
                            policy_ys_std = np.std(policy_y, 0)
                            #policy_rs_mean = np.mean(advantages, 0)
                            #policy_rs_std = np.std(advantages, 0)
                        policy_X = standardise(policy_X, policy_Xs_mean, policy_Xs_std)
                        policy_y = standardise(policy_y, policy_ys_mean, policy_ys_std)
                        """
                        # train policy
                        policy.train(policy_X, policy_y, advant, POLICY_TRAINING_LOG_FILE)

                        # Empty policy training data buffer
                        policy_Xs = []
                        policy_ys = []
                        advantages = []
                        policy_buff_size = 0
                # Reset the scene
                current_state = env.reset(False)[np.newaxis, :]

                # From episode SWITCH_POINT_EP onward switch between sampling from model and sampling from real environment
                if i >= SWITCH_POINT_EP:
                    SAMPLE_FROM_MODEL = not SAMPLE_FROM_MODEL

            training_wall_clock_time = time.time() - training_start_time
            average_last_20_reward_sum = sum(reward_sums) / len(reward_sums)
            if ADDITIONAL_LOGGING:
                with open(SESSION_ROOT + '/' + TRAINING_SUMMARY_FILE, 'a') as f:
                    f.write("{}, {}, {}, {}\n".format(SESSION_ID, num_training_steps, training_wall_clock_time,
                        average_last_20_reward_sum))
            if NEW_MODEL:
                model.save(MODEL_FILE)
            if NEW_POLICY:
                policy.save()
