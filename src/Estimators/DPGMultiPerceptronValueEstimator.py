import tensorflow as tf
import tflearn

class DPGMultiPerceptronValueEstimator(object):
    """ Deterministic Value Gradient Policy Function Approximator

    The estimator in use deploys a multi-layer perceptron

    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    NB: This implementation assumes a particular architecture, thus unchangeable, only with tunable
    hyper-parameters
    """

    def __init__(self, sess, state_dim, action_dim, h1_dim, h2_dim, learning_rate, tau, num_actor_vars,
    scope="value_estimator"):
        self._sess = sess
        self._s_dim = state_dim
        self._a_dim = action_dim
        self._h1_dim = h1_dim
        self._h2_dim = h2_dim
        self._learning_rate = learning_rate
        self._tau = tau

        # Create the critic network
        self._inputs, self._action, self._out = self._create_critic_network(scope)

        self._network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self._target_inputs, self._target_action, self._target_out = self._create_critic_network(scope + "_target")

        self._target_network_params = tf.trainable_variables()[(len(self._network_params) + num_actor_vars):]

        with tf.name_scope(scope + "_target/"):
            # Op for periodically updating target network with online network
            # weights with regularization
            self._update_target_network_params = \
                [self._target_network_params[i].assign(tf.multiply(self._network_params[i], self._tau) \
                + tf.multiply(self._target_network_params[i], 1. - self._tau))
                    for i in range(len(self._target_network_params))]

        with tf.name_scope(scope):
            # Network target (y_i)
            self._predicted_q_value = tf.placeholder(tf.float32, [None, 1], name="predicted_q_value")
            # Define loss and optimization Op
            self._loss = tflearn.mean_square(self._predicted_q_value, self._out)
            self._optimize = tf.train.AdamOptimizer(
                self._learning_rate).minimize(self._loss, name="optimize")

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
            #self._predicted_q_value = graph.get_tensor_by_name(scope + "/predicted_q_value:0")
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
            inputs = tflearn.input_data(shape=[None, self._s_dim], name="inputs")
            action = tflearn.input_data(shape=[None, self._a_dim], name="action")
            net = tflearn.fully_connected(inputs, self._h1_dim)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            t1 = tflearn.fully_connected(net, self._h2_dim)
            t2 = tflearn.fully_connected(action, self._h2_dim)

            net = tflearn.activation(
                tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(net, 1, weights_init=w_init, name="out")
        return inputs, action, out

    def update(self, inputs, action, predicted_q_value):
        return self._sess.run([self._out, self._optimize, self._loss], feed_dict={
            self._inputs: inputs,
            self._action: action,
            self._predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self._sess.run(self._out, feed_dict={
            self._inputs: inputs,
            self._action: action
        })

    def predict_target(self, inputs, action):
        return self._sess.run(self._target_out, feed_dict={
            self._target_inputs: inputs,
            self._target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self._sess.run(self._action_grads, feed_dict={
            self._inputs: inputs,
            self._action: actions
        })

    def update_target_network(self):
        self._sess.run(self._update_target_network_params)
