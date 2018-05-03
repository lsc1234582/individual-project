import tensorflow as tf
import tflearn

class MultiPerceptronModelEstimator(object):
    """ Deterministic Value Gradient Model Approximator

    The estimator in use deploys a multi-layer perceptron

    Input to the network is the state and action, output is the predicted change in states
    Inputs and outputs are normalised to have 0 mean and std of 1
    """

    def __init__(self, sess, state_processor, state_dim, action_dim, h_layer_shapes, learning_rate,
    scope="model_estimator"):
        self._sess = sess
        self._state_processor = state_processor
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._h_layer_shapes = h_layer_shapes
        self._learning_rate = learning_rate

        # Create the model network
        self._inputs, self._action, self._outputs = self._create_model_network(scope)

        with tf.name_scope(scope):
            # Network target (y_i)
            self._actual_outputs = tf.placeholder(tf.float32, [None, self._state_dim], name="actual_outputs")
            # Define loss and optimization Op
            self._loss = tflearn.mean_square(self._actual_outputs, self._outputs)
            self._optimize = tf.train.AdamOptimizer(
                self._learning_rate).minimize(self._loss, name="optimize")

            #graph = tf.get_default_graph()

            #self._inputs = graph.get_tensor_by_name(scope + "/inputs:0")
            #self._action = graph.get_tensor_by_name(scope + "/action:0")
            #self._outputs = graph.get_tensor_by_name(scope + "/out:0")
            #self._actual_outputs = graph.get_tensor_by_name(scope + "/predicted_q_value:0")
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

    def _create_model_network(self, scope):
        with tf.name_scope(scope):
            inputs = tflearn.input_data(shape=(None, self._state_dim), name="inputs")
            action = tflearn.input_data(shape=(None, self._action_dim), name="action")

            net = tf.concat([inputs, action], axis=1)

            for i in range(len(self._h_layer_shapes) - 1):
                net = tflearn.fully_connected(net, self._h_layer_shapes[i])
                net = tflearn.layers.normalization.batch_normalization(net)
                net = tflearn.activations.relu(net)

            # linear layer
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            outputs = tflearn.fully_connected(net, self._state_dim, weights_init=w_init, name="outputs")
        return inputs, action, outputs

    def update(self, inputs, action, actual_outputs):
        inputs = self._state_processor.transform(inputs)
        actual_outputs = self._state_processor.transform(actual_outputs)
        return self._sess.run([self._outputs, self._optimize, self._loss], feed_dict={
            self._inputs: inputs,
            self._action: action,
            self._actual_outputs: actual_outputs
        })

    def predict(self, inputs, action):
        outputs = self._sess.run(self._outputs, feed_dict={
            self._inputs: inputs,
            self._action: action
        })
        return self._state_processor.inverse_transform(outputs)

    def evaluate(self, inputs, action, actual_outputs):
        inputs = self._state_processor.transform(inputs)
        actual_outputs = self._state_processor.transform(actual_outputs)
        return self._sess.run([self._loss], feed_dict={
            self._inputs: inputs,
            self._action: action,
            self._actual_outputs: actual_outputs
            })
