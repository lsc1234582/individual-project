import collections
import gzip
import logging
import random
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
from collections import deque
from Utils.SegmentTree import SumSegmentTree, MinSegmentTree

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def featurize_state(state, scaler):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform(state)
    return scaled

def get_scalar(var):
    if type(var) == np.ndarray:
        assert(var.shape == (1,))
        return var[0]
    elif type(var) == list or type(var) == tuple:
        assert(np.array(var).shape == 1)
        return var[0]
    return var

def getModuleLogger(module_name):
    """
    Universal configuration of logger across all modules.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Create handlers
    fh = logging.FileHandler("run_info.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    fdh = logging.FileHandler("run_debug.log")
    fdh.setLevel(logging.DEBUG)
    fdh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(fh)
    logger.addHandler(fdh)
    logger.addHandler(ch)
    return logger

def generateRandomAction(max_vel, dim=6):
    """
        Generate an array of shape (dim,) of range [-max_vel, max_vel].
    """
    return np.array([random.random() * max_vel * 2 - max_vel for _ in range(dim)])

class SortedDisplayDict(dict):
   def __str__(self):
       return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self)) + "}"

class ReplayBuffer(object):
    """
    Modefied based on OpenAI baseline implementation:

    url: https://github.com/openai/baselines
    author: Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert,
    Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai
    year: 2017
    """
    def __init__(self, size, debug=False):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._is_bkup = False
        self._debug = debug
        # _debug_freq is a list of frequencies; its indexes correspond to the indexes of the replay buffer
        if self._debug:
            self._debug_freq = []
        self._stats_estimate_sample_size = int(1e4)

    def __eq__(self, other):
        if not (self._maxsize == other._maxsize and len(self) == len(other) and self._next_idx == other._next_idx \
                 and self._stats_estimate_sample_size == other._stats_estimate_sample_size):
            return False
        for i in range(len(self)):
            for j in range(len(self._storage[i])):
                if np.any(self._storage[i][j] != other._storage[i][j]):
                    return False
        return True

    def _do_load(self, f):
        states = pkl.load(f)
        if "_storage" in states:
            self._storage = states["_storage"]
        else:
            self._storage = states["_buffer"]
        if "_maxsize" in states:
            self._maxsize = states["_maxsize"]
        else:
            self._maxsize = states["_buffer_size"]
        if "_next_idx" in states:
            self._next_idx = states["_next_idx"]
        else:
            self._next_idx = states["_count"]
        self._is_bkup = states["_is_bkup"]
        self._stats_estimate_sample_size = states["_stats_estimate_sample_size"]
        if self._debug:
            self._debug_freq = [0 for _ in self._storage]

        return states

    def load(self, load_path):
        if not os.path.exists(os.path.join(load_path, "checkpoint")):
            raise IOError("No replay buffer checkpoint found")
        with open(os.path.join(load_path, "checkpoint"), "r") as f:
            rb_file_name = f.read()
        with gzip.open(os.path.join(load_path, rb_file_name), "rb") as f:
            self._do_load(f)

    def save(self, save_path):
        """
        Use double buffer to provide a safe backup (if pickling is interrupted the save file is permenantly damaged)
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self._is_bkup:
            rb_file_name =  "replay-buffer-0.pklz"
        else:
            rb_file_name =  "replay-buffer-1.pklz"

        rb_save_path = os.path.join(save_path, rb_file_name)

        with gzip.open(rb_save_path, "wb") as f:
            pkl.dump(vars(self), f)

        self._is_bkup = not self._is_bkup
        with open(os.path.join(save_path, "checkpoint"), "w") as f:
            f.write(rb_file_name)


    def clear(self):
        self._storage.clear()
        if self._debug:
            self._debug_freq.clear()
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def size(self):
        return len(self)

    def add(self, obs_t, action, reward, obs_tp1, done, goal=None):
        if goal is None:
            data = (obs_t, action, reward, obs_tp1, done)
        else:
            data = (obs_t, action, reward, obs_tp1, done, goal)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            if self._debug:
                self._debug_freq.append(0)
        else:
            self._storage[self._next_idx] = data
            if self._debug:
                self._debug_freq[self._next_idx] = 0

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """

        Returns
        -------

        All in shape (batch_size, dim)
        """
        obses_t, actions, rewards, obses_tp1, dones, goals = [], [], [], [], [], []
        with_goal = False
        for i in idxes:
            data = self._storage[i]
            if len(data) == 5:
                obs_t, action, reward, obs_tp1, done = data
            elif len(data) == 6:
                obs_t, action, reward, obs_tp1, done, goal = data
                with_goal = True
            else:
                assert False
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            if len(data) == 6:
                goals.append(np.array(goal, copy=False))

        if with_goal:
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones),\
                    np.array(goals)
        else:
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _retrieve_eps_from_transition(self, rand_idx):
        """ Retrieve a complete episode indexes which includes rand_idx.
        Assumes the replay buffer stores the transition data in order.
        NOTE:
            The retrieved indexes always include the true start but not necessarily the true end. The end returned by
            this method may be the true end, and it may also be the end of the replay buffer, which is not necessarily
            the end of an episode. In other words, this method may return incomplete episode transitions towards the
            end.
        """
        assert 0 <= rand_idx < len(self._storage)

        # Locate the first transition from the episode which includes rand_idx transition
        start_idx = rand_idx - 1
        _, _, _, _, done = self._storage[start_idx]
        while not get_scalar(done) and start_idx >= 0:
            start_idx -= 1
            _, _, _, _, done = self._storage[start_idx]
        start_idx += 1

        # Locate the one past last transition from the episode which includes rand_idx transition
        end_idx = rand_idx
        _, _, _, _, done = self._storage[end_idx]
        while not get_scalar(done) and end_idx < len(self._storage) - 1:
            end_idx += 1
            _, _, _, _, done = self._storage[end_idx]
        _, _, _, _, is_complete = self._storage[end_idx]
        end_idx += 1
        return list(range(start_idx, end_idx)), get_scalar(is_complete)

    def sample_episode(self):
        """Sample a complete episode rollout.
        """
        rand_idx = random.randint(0, len(self._storage) - 1)

        idxes, is_complete = self._retrieve_eps_from_transition(rand_idx)
        return tuple(list(self._encode_sample(idxes)) + [is_complete])

    def sample_batch(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        if self._debug:
            for idx in idxes:
                self._debug_freq[idx] += 1

        return self._encode_sample(idxes)

    def debug_get_sample_freq(self):
        assert self._debug
        return self._debug_freq


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Modefied based on OpenAI baseline implementation:

    url: https://github.com/openai/baselines
    author: Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert,
    Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai
    year: 2017
    """
    def __init__(self, size, alpha, debug=False):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, debug)
        assert alpha >= 0
        self._alpha = alpha

        self._calculateCapacity(size)

        self._it_sum = SumSegmentTree(self._it_capacity)
        self._it_min = MinSegmentTree(self._it_capacity)
        self._max_priority = 1.0

        # Store the size of loaded content, in order to separate from loaded episodes (potentially demo data) and live
        # episodes.
        self._loaded_storage_size = 0

    def _calculateCapacity(self, size):
        self._it_capacity = 1
        while self._it_capacity < size:
            self._it_capacity *= 2


    def __eq__(self, other):
        return super().__eq__(other) and (self._it_sum == other._it_sum and self._it_min == other._it_min\
                and self._max_priority == other._max_priority)


    def clear(self):
        super(PrioritizedReplayBuffer, self).clear()
        del self._it_sum, self._it_min
        self._it_sum = SumSegmentTree(self._it_capacity)
        self._it_min = MinSegmentTree(self._it_capacity)
        self._max_priority = 1.0

    def _do_load(self, f):
        states = super(PrioritizedReplayBuffer, self)._do_load(f)

        new_seg_tress = False
        self._calculateCapacity(self._maxsize)
        if "_alpha" in states:
            self._alpha = states["_alpha"]
        if "_it_sum" in states:
            self._it_sum = states["_it_sum"]
        else:
            self._it_sum = SumSegmentTree(self._it_capacity)
            new_seg_tress = True
        if "_it_min" in states:
            self._it_min = states["_it_min"]
        else:
            self._it_min = MinSegmentTree(self._it_capacity)
            new_seg_tress = True
        if "_max_priority" in states:
            self._max_priority = states["_max_priority"]
        if "_it_capacity" in states:
            self._it_capacity = states["_it_capacity"]

        self._loaded_storage_size = len(self._storage)
        # Initialise segment trees to contain max_priority initially
        for i in range(self._loaded_storage_size):
            self._it_sum[i] = self._max_priority ** self._alpha
            self._it_min[i] = self._max_priority ** self._alpha

        return states

    def get_loaded_storage_size(self):
        return self._loaded_storage_size

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect

        Parameters
        ----------

        priority:       int         Positive. priority of an associated experience.
        """
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size, no_repeat=True):
        ps = []
        res = []
        for i in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            # TODO: Unbalanaced dataset?
            #print(self._it_sum._value)
            #mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            #mass = random.random() * self._it_sum.sum()
            mass = (i / batch_size + random.random() / batch_size) * self._it_sum.sum()
            idx = self._it_sum.find_prefixsum_idx(mass)
            if no_repeat:
                # Set priority at idx to 0 to ensure no repeats
                ps.append(self._it_sum[idx])
                self._it_sum[idx] = 0.0
            res.append(idx)

        if no_repeat:
            # Restore original priority
            for idx, original_p in zip(res, ps):
                self._it_sum[idx] = original_p
        return res

    def sample_batch(self, batch_size, beta, no_repeat=True):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample_batch
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0
        assert len(self._storage) >= batch_size

        idxes = self._sample_proportional(batch_size, no_repeat=no_repeat)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
            if self._debug:
                self._debug_freq[idx] += 1
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def sample_episode(self, beta):
        """Sample a complete episode rollout
        """
        assert beta > 0

        rand_idx = self._sample_proportional(1)[0]

        idxes, is_complete = self._retrieve_eps_from_transition(rand_idx)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
            if self._debug:
                self._debug_freq[idx] += 1
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes, is_complete])


    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def debug_get_priorities(self):
        """
        Returns [attenuated] priorities
        """
        assert self._debug
        return self._it_sum[:len(self._storage)]

class SummaryWriter(object):

    def __init__(self, sess, summary_dir):
        self._sess = sess
        self._summary_vars = None
        self._writer = tf.summary.FileWriter(summary_dir, sess.graph)


    def writeSummary(self, var_values, training_step):
        if self._summary_vars is None:
            self._summary_vars = {}
            with tf.variable_scope("Stats_Summary"):
                for var_name in var_values:
                    self._summary_vars[var_name] = tf.get_variable(shape=(), trainable=False, name=var_name)
                    tf.summary.scalar(var_name, self._summary_vars[var_name])
                self._summary_ops = tf.summary.merge_all()
        else:
            assert(self._summary_vars.keys() == var_values.keys())
        feed_dict = {self._summary_vars[var_name]: var_values[var_name] for var_name in var_values.keys()}
        summary_str = self._sess.run(self._summary_ops, feed_dict=feed_dict)
        self._writer.add_summary(summary_str, training_step)
        self._writer.flush()

class OrnsteinUhlenbeckActionNoise(object):
    """
    Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
    based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

if __name__ == "__main__":
    pass
