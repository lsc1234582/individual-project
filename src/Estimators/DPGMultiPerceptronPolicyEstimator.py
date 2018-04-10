import tensorflow as tf
import tflearn

class DPGMultiPerceptronPolicyEstimator(object):
    """ Deterministic Policy Gradient Policy Function Approximator

    The estimator in use deploys a multi-layer perceptron

    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, layer_shapes, learning_rate, action_bound, tau, minibatch_size):
        self._sess = sess
        self._layer_shapes = layer_shapes
        self._action_bound = action_bound
        self._learning_rate = learning_rate
        self._tau = tau
        self._minibatch_size = minibatch_size

        # Actor Network
        self._inputs, self._out, self._scaled_out = self._create_actor_network()

        self._network_params = tf.trainable_variables()

        # Target Network
        self._target_inputs, self._target_out, self._target_scaled_out = self._create_actor_network()

        self._target_network_params = tf.trainable_variables()[
            len(self._network_params):]

        # Op for periodically updating target network with online network
        # weights
        self._update_target_network_params = \
            [self._target_network_params[i].assign(tf.multiply(self._network_params[i], self._tau) +
                                                  tf.multiply(self._target_network_params[i], 1. - self._tau))
                for i in range(len(self._target_network_params))]

        # This gradient will be provided by the critic network
        self._action_gradient = tf.placeholder(tf.float32, [None, self._layer_shapes[-1]])

        # Combine the gradients here
        self._unnormalized_actor_gradients = tf.gradients(
            self._scaled_out, self._network_params, -self._action_gradient)
        self._actor_gradients = list(map(lambda x: tf.div(x, self._minibatch_size), self._unnormalized_actor_gradients))

        # Optimization Op
        self._optimize = tf.train.AdamOptimizer(self._learning_rate).\
            apply_gradients(zip(self._actor_gradients, self._network_params))

        self._num_trainable_vars = len(
            self._network_params) + len(self._target_network_params)

    def _create_actor_network(self):
        inputs = tflearn.input_data(shape=(None, self._layer_shapes[0]))
        net = inputs
        for i in range(1, len(self._layer_shapes) - 1):
            net = tflearn.fully_connected(net, self._layer_shapes[i])
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self._layer_shapes[-1], activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self._action_bound)
        return inputs, out, scaled_out

    def update(self, inputs, a_gradient):
        tflearn.is_training(True, self._sess)
        self._sess.run(self._optimize, feed_dict={
            self._inputs: inputs,
            self._action_gradient: a_gradient
        })

    def predict(self, inputs):
        tflearn.is_training(False, self._sess)
        return self._sess.run(self._scaled_out, feed_dict={
            self._inputs: inputs
        })

    def predict_target(self, inputs):
        return self._sess.run(self._target_scaled_out, feed_dict={
            self._target_inputs: inputs
        })

    def update_target_network(self):
        self._sess.run(self._update_target_network_params)

    def get_num_trainable_vars(self):
        return self._num_trainable_vars
