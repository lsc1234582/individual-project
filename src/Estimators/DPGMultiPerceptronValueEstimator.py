import tensorflow as tf
import tflearn
from Utils import normalize, denormalize

class DPGMultiPerceptronValueEstimator(object):
    """ Deterministic Value Gradient Value Function Approximator

    The estimator in use deploys a multi-layer perceptron

    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    NB: This implementation assumes a particular architecture
    """

    def __init__(self, sess, state_rms, return_rms, state_dim, action_dim, h_layer_shapes, state_range, return_range,
            learning_rate, tau, num_actor_vars, lambda2=0.1,
            scope="value_estimator"):
        self._sess = sess
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._h_layer_shapes = h_layer_shapes
        self._state_range = state_range
        self._return_range = return_range
        self._learning_rate = learning_rate
        self._tau = tau
        # Weights of the n-step term in the loss function
        # TODO: remove hardcoded.
        # NB: lambda2 = 1/num_steps for n-step td calculation
        self._lambda2 = lambda2

        self._state_rms = state_rms
        self._return_rms = return_rms

        # Create the critic network
        self._inputs, self._action, self._out = self._create_critic_network(scope)

        # Denormalized out
        self._denorm_out = denormalize(tf.clip_by_value(self._out, self._return_range[0],
                    self._return_range[1]), self._return_rms)

        self._network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self._target_inputs, self._target_action, self._target_out = self._create_critic_network(scope + "_target")

        # Denormalized target out
        self._denorm_target_out = denormalize(tf.clip_by_value(self._target_out, self._return_range[0],
                    self._return_range[1]), self._return_rms)

        self._target_network_params = tf.trainable_variables()[(len(self._network_params) + num_actor_vars):]

        # Make tau a placeholder so we can plug in different values within this class
        self._tau_var = tf.placeholder(tf.float32, [], name="tau")
        with tf.name_scope(scope + "_target/"):
            # Op for periodically updating target network with online network
            # weights with regularization
            self._update_target_network_params = \
                [self._target_network_params[i].assign(tf.multiply(self._network_params[i], self._tau_var) \
                + tf.multiply(self._target_network_params[i], 1. - self._tau_var))
                    for i in range(len(self._target_network_params))]

        with tf.name_scope(scope):
            opt = tf.train.AdamOptimizer(self._learning_rate)
            # Network td_target
            self._td_target = tf.placeholder(tf.float32, [None, 1], name="td_target")

            self._normalized_td_target = tf.clip_by_value(normalize(self._td_target, self._return_rms), self._return_range[0],
                    self._return_range[1])
            # Define loss and optimization Op
            # NB: self._out is defined to be normalized
            # TODO: denormalize _td_error for outputing?
            self._td_error = self._normalized_td_target - self._out
            td_error_sq = tf.square(self._td_error)
            self._loss = tf.reduce_mean(td_error_sq)
            self._optimize = opt.minimize(self._loss, var_list=self._network_params, name="optimize")

            # Weighted update
            self._update_weights = tf.placeholder(tf.float32, [None, 1], name="update_weights")
            weighted_td_error_sq = self._update_weights * td_error_sq
            self._weighted_loss = tf.reduce_mean(weighted_td_error_sq)

            self._weighted_optimize = opt.minimize(self._weighted_loss, var_list=self._network_params, name="weighted_optimize")

            # Weighted n-1 step update
            # The loss function is defined in the paper [Leverage...]
            self._nb_ns_td_target = tf.placeholder(tf.int32, [], name="nb_ns_td_target")
            mid_indx = tf.shape(td_error_sq)[0] - self._nb_ns_td_target
            self._n1s_loss = tf.reduce_mean(td_error_sq[:mid_indx])\
                    + self._lambda2 * tf.reduce_mean(td_error_sq[mid_indx:])
            self._weighted_n1s_loss = tf.reduce_mean(weighted_td_error_sq[:mid_indx])\
                    + self._lambda2 * tf.reduce_mean(weighted_td_error_sq[mid_indx:])
            self._weighted_n1s_optimize = opt.minimize(self._weighted_n1s_loss, var_list=self._network_params, name="weighted_n1s_optimize")

            # Get the gradient of the net w.r.t. the action.
            # For each action in the minibatch (i.e., for each x in xs),
            # this will sum up the gradients of each critic output in the minibatch
            # w.r.t. that action. Each output is independent of all
            # actions except for one.
            self._action_grads = tf.gradients(self._out, self._action, name="action_grads")

            #graph = tf.get_default_graph()

            #self._inputs = graph.get_tensor_by_name(scope + "/inputs:0")
            #self._action = graph.get_tensor_by_name(scope + "/action:0")
            #self._out = graph.get_tensor_by_name(scope + "/out:0")
            #self._td_target = graph.get_tensor_by_name(scope + "/predicted_q_value:0")
            #self._loss = graph.get_tensor_by_name(scope + "/loss:0")
            #self._optimize = graph.get_tensor_by_name(scope + "/optimize:0")
            #self._action_grads = graph.get_tensor_by_name(scope + "/action_grads:0")
            #self._target_inputs = graph.get_tensor_by_name(scope + "_target/inputs:0")
            #self._target_action = graph.get_tensor_by_name(scope + "_target/action:0")
            #self._target_out = graph.get_tensor_by_name(scope + "_target/out:0")

            #self._network_params = tf.trainable_variables()[num_actor_vars:]
            #self._target_network_params = tf.trainable_variables()[(len(self._network_params) + num_actor_vars):]
            #self._update_target_network_params = []
            #for i in range(len(self._target_network_params)):
            #    self._update_target_network_params[i] = graph.get_tensor_by_name(
            #            scope+"_target/update_target_network_params:{}".format(i))

    def _create_critic_network(self, scope):
        with tf.name_scope(scope):
            inputs = tflearn.input_data(shape=(None, self._state_dim), name="inputs")
            action = tflearn.input_data(shape=(None, self._action_dim), name="action")
            # Normalized input(states)
            normalized_inputs = tf.clip_by_value(normalize(inputs, self._state_rms), self._state_range[0],
                    self._state_range[1])

            net = normalized_inputs

            for i in range(len(self._h_layer_shapes) - 1):
                net = tflearn.fully_connected(net, self._h_layer_shapes[i])
                # Add l2 regularizer
                tflearn.helpers.regularizer.add_weights_regularizer(net.W, "L2")
                net = tflearn.layers.normalization.batch_normalization(net)
                net = tflearn.activations.relu(net)

            # Add the action tensor in the last hidden layer
            # Use two temp layers to get the corresponding weights and biases
            t1 = tflearn.fully_connected(net, self._h_layer_shapes[-1])
            tflearn.helpers.regularizer.add_weights_regularizer(t1.W, "L2")
            t2 = tflearn.fully_connected(action, self._h_layer_shapes[-1])
            tflearn.helpers.regularizer.add_weights_regularizer(t2.W, "L2")

            net = tflearn.activation(
                tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(net, 1, weights_init=w_init, name="out")
            out = tf.clip_by_value(normalize(out, self._return_rms), self._return_range[0],
                    self._return_range[1])
        return inputs, action, out

    def update(self, inputs, action, predicted_q_value):
        return self._sess.run([self._optimize, self._loss], feed_dict={
            self._inputs: inputs,
            self._action: action,
            self._td_target: predicted_q_value
        })

    def update_with_weights(self, inputs, action, td_target, weights):
        return self._sess.run([self._weighted_optimize, self._td_error, self._weighted_loss, self._loss], feed_dict={
            self._inputs: inputs,
            self._action: action,
            self._td_target: td_target,
            self._update_weights: weights
        })

    def update_with_weights_and_n1s_td(self, inputs, action, td_target, weights, nb_ns_td_target):
        """
        Args
        -----
        inputs: Combined 1-step and n-step batches, where the [-nb_ns_td_target:] are n-step batches
        action: Combined 1-step and n-step batches, where the [-nb_ns_td_target:] are n-step batches
        td_target: ...
        weights: ...
        nb_ns_td_target: Number of N-step td targets in the batch
        """
        return self._sess.run([self._weighted_n1s_optimize, self._td_error, self._weighted_n1s_loss,\
            self._n1s_loss], feed_dict={
            self._inputs: inputs,
            self._action: action,
            self._td_target: td_target,
            self._update_weights: weights,
            self._nb_ns_td_target: nb_ns_td_target
        })

    def predict(self, inputs, action):
        return self._sess.run(self._denorm_out, feed_dict={
            self._inputs: inputs,
            self._action: action
        })

    def predict_target(self, inputs, action):
        return self._sess.run(self._denorm_target_out, feed_dict={
            self._target_inputs: inputs,
            self._target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self._sess.run(self._action_grads, feed_dict={
            self._inputs: inputs,
            self._action: actions
        })

    def update_target_network(self, tau=None):
        if tau == None:
            tau = self._tau
        self._sess.run(self._update_target_network_params, feed_dict={
            self._tau_var: tau
            })
