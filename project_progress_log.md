16/01/2018

Finished setting up the model-based learning for push task. Incorporating the additional two vectors
(gripper-to-cuboid-vector and cuboid-to-target-plane-vector) deterioriated the training of the model  dramatically. The
extra hidden layer wouldn't help.

The cost/neg reward is simply the sum of the euclidean distance between cuboid and gripper and target and cuboid.
Investigate different reward function (TODO)

The result was somewhat decent; the reaching part was almost always perfectly executed but the pushing part not quite so. In addition MPC is way too slow at this point. Consider switching to an explicity policy (TODO).

Came across [a series of good
tutorials](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99)
which include model-based learning (TODO current progress Part3). Right now the reward is an explicit function while it
probably should be part of the model (TODO Investigate).

Also consider adding a stopping condition(destination) to the episodes (e.g. when the cuboid is mostly inside the
target plane or the cuboid is outside the reach of the arm) (TODO)


02/02/2018

Finished tasks:
1. Implement policy-gradient method (somewhat).
2. Abstract away the communication with VREP simulator; the simulation is presented as an environment.

The policy is modeled as a Gaussian distribution, with the
output of a neural net as the "mean action" and an identity covariance matrix. The neural net accepts states as input.
The _advantage_ of a step within a trajectory is simply the sample return (from this step till the end of episode), or
*reward_to_go*, minus the baseline, average return over all steps of the trajectory. However because the *reward_to_go*
is estimated using a single sample, the variance is very high, resulting in slow or even bad convergence
(https://youtu.be/PpVhtJn-iZI?t=4m28s).

The results so far are not great. There are very little to no learning. I also encountered exploding gradient problems
during training. This suggests that I need to undergo a lot of hyperparameter tuning (especially learning rate) and
data conditioning.

Another solution which reduces the variance is to use an actor-critic method, which uses value function to estimate the
_advantage_ (https://youtu.be/PpVhtJn-iZI?t=29m50s). This approach just needs a little bit of tweaking of the current
method, but requires training another neural net for approximating the value function. This should in principle leads
to a more stable and faster training process.

However even actor-critic is still not very sample efficient (requires a lot of samples to learn a good value function). And
the dynamics model is not used anywhere. The only usage of the model seems to be for sampling simulated trajectories.
Another direction is to forgo the current method and uses something like **Guided Policy Search**. The general idea is
to learn a fairly accurate dynamics model first, then use the model and some sampled trajectories to construct a set of
optimal trajectories using trajectory optimisation techniques such as iLQR and finally use supervise learning to train
the policy on these optimal trajectories (**Optimal imitation learning**). These concepts are explained in more details
in these three videos, (https://youtu.be/EfgC7v5V608) (https://youtu.be/yap_g0d7iBQ) (https://youtu.be/AwdauFLan7M) and
this paper (Learning contact-rich manipulation skills with guided policy search). The advantages of this approach
include that it's much more sample efficient and fully makes use of the learned model, and that it should lead to more
stable training process, since imitation learning is more well-understood and simpler than policy gradient.

Currently I have three options:
1. Stick to policy gradient and tune the policy.
2. Switch to actor-critic.
3. Switch to fully model-based learning like GPS.

Each is more sample-efficient and more stable than the last, but each requires
more research and development time than the last. For now I want to explore a bit more about GPS because I think it
will benefit when the controller is migrated to the real robot, and also I want to have a more solid foundation at the
current full-state RL stage, considering policy gradient is generally an unstable method.

More TODOs:
1. Clean up repo and share with Ed.
2. Investigate and implement GPS.
3. Consider using Pandoc for generating this log book so I can embed Latex.

07/02/2018


1. Cleaned up repository (somewhat) by having separate module files for VREPMicoPushTaskEnvironment and GaussianPolicy.
2. Fixed broken context manager associated with VREP environment. Now all connections and resources are freed properly
when the training process halts for whatever reason.
3. Fixed some fatal bugs in VREPMicoPushTaskEnvironment such as the internal state is not advancing with the step()
   function. Previous poor training results largely resulted from this.
4. Added constraints on the collected trajectory to avoid bad training data by setting a limit on allowed max Mico joint velocity.
5. Added a logistic function as greedy factor scheduler. (Currently the greedy factor caps at 0.95; need to explore
   more later during hyper parameter tuning)
6. Added a swapping mechanism for interaction with the simulator and model. Now before 100th episode the samples are
   drawn from simulator only; after 100th episode they are drawn from simulator and trained model alternately.
7. Added callbacks for checkpointing models, logging metrics (training and validation losses for model, training losses
   for policy and summed rewards for each episode)
8. Restructured how the training process data are stored by organising them into different training sessions. It will
   be easier for later hyper parameter tuning and documenting.
9. Updated README

TODOs:
The main task right now is to train a controller with PolicyGradient till it gets good result, then decide if we should
try a different method like ActorCritic or GPS or something else based on how this optimally trained controller
performs the task and how many samples as well as how much wall-clock trianing time are put into this.

Before training such a controller we will first need to find a good set of parameters. In preparation for parameter
tuning, I still need to tie a few loose ends on Thursday.
1. The current controllers are not trained properly due to constant debugging and code change. Training a controller from start to finish and try to find more bugs and fix them before moving on.
2. Abstract the training sessions. Move them out of the notebook and into modular python script files3.
3. Investigate launching multiple VREP instances. It could accelerate the parameter tuning process.
4. Keep cleaning up the repo and updating README.
