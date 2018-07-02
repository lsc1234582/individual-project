# Deep Reinforcement Learning for Robotics: An Evaluation of Models, Rewards and Demonstrations

## Prerequisite
1. VREP
2. Python >= 3.4
3. Pip >= 9.0.1

To install all dependencies required, run:
```
pip install -r requirements
```

## Steup
The remote API server is enabled at VREP start-up. To enable it, change the remoteApiConnections.txt with the following
settings: portIndex1_port to 19997, and portIndex1_syncSimTrigger.
The details can be found in VREP document [Enabling the Remote API - server side](http://www.coppeliarobotics.com/helpFiles/en/remoteApiServerSide.htm) and [Enabling the Remote API - client side](http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm).

The program for DDPGfD agent is named DPGAC2WithPrioritizedRBPEH2VEH2.py, and the one for model-based agent is named
ModelBasedMEH2.py

The available environments are:
1. VREPPushTask7DoF (Push + Dense)
2. VREPPushTask7DoFSparseRewards (Push + Sparse)
3. VREPGraspTask7DoF (Grasp + Dense)
4. VREPGraspTask7DoFSparseRewards (Grasp + Sparse)

For example, to train a DDPGfD agent in the Push task, first launch the simulation:
```
vrep.sh VREPScene/MicoPush.ttt
```
Then start the program by specifying the environment name, the directories for estimators and training stats summary:
```
python src/DPGAC2WithPrioritizedRBPEH2VEH2.py --env-name VREPPushTask7DoF --estimator-dir ./estimator --summary-dir summary --new-estimator
```

For a full list of hyper-parameter arguments and other options run:
```
python src/DPGAC2WithPrioritizedRBPEH2VEH2.py -h
```

