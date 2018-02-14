import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import BatchNormalization, Dense, Input, Lambda
from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

class GaussianPolicy:
    def reinforcementLoss(args):
        action_mean, action_taken, advantages, action_cov = args
        """
            Return a a Refinforcement loss function to be optimised
        """
        at_minus_am = action_taken - action_mean
        loglike = at_minus_am[:, :, tf.newaxis] * action_cov * at_minus_am[:, tf.newaxis, :]
        loglike = -0.5 * tf.reduce_mean(loglike, axis=[1, 2])
        #loglike = -0.5 * tf.matmul(tf.square(action_taken - action_mean), tf.matrix_inverse(action_cov))
        return -tf.reduce_mean(loglike * advantages)

    """
    self.action(6,) is sampled from self.policy,
    which is a gaussian distribution with mean as the output from self.model
    self.model takes state (24,) and produces an action mean (6,)
    TODO: Constrain self.model output?
    """
    def __init__(self, model_file, lr=0.05, epochs=1, batch_size=128, seed=0, is_load_model=False):
        if not is_load_model:
            #K.set_learning_phase(True)
            self.state = Input(shape=(24,), dtype="float32", name="state")
            hidden_l1 = Dense(256, kernel_initializer="random_normal", kernel_regularizer=keras.regularizers.l2(0.01),
                    bias_initializer="zeros", activation="relu", name="hidden_l1")(self.state)
            hidden_l1_bn = BatchNormalization(name="hidden_l1_bn")(hidden_l1)
            hidden_l2 = Dense(64, kernel_initializer="random_normal", kernel_regularizer=keras.regularizers.l2(0.01),
                    bias_initializer="zeros", activation="relu", name="hidden_l2")(hidden_l1_bn)
            hidden_l2_bn = BatchNormalization(name="hidden_l2_bn")(hidden_l2)
            self.action_mean = Dense(6, kernel_initializer="random_normal", kernel_regularizer=keras.regularizers.l2(0.01),
                    bias_initializer="zeros", name="action_mean_l")(hidden_l2_bn)
            # Additional inputs for calculating reinforcementLoss
            self.action_taken = Input(shape=(6,), dtype="float32", name="action_taken")
            self.advantage = Input(shape=(1,), dtype="float32", name="advantage")
            self.action_cov = Input(shape=(6,6,), dtype="float32", name="action_cov")
            loss = Lambda(GaussianPolicy.reinforcementLoss, output_shape=(1,), name="reinforcement_loss")(\
                    [self.action_mean, self.action_taken, self.advantage, self.action_cov])
            self.model = Model(inputs=[self.state, self.action_taken, self.advantage, self.action_cov], outputs=loss)
            # Use a dummy loss for Keras' internal loss
            self.model.compile(loss={'reinforcement_loss': lambda action_taken, action_mean: action_mean},
                    optimizer='rmsprop')
        else:
            # load an existing model
            self.model = load_model(model_file, custom_objects={'reinforcement_loss': lambda action_taken, action_mean:
                action_mean})
            self.state = self.model.inputs[0]
            self.action_taken = self.model.inputs[1]
            self.advantage = self.model.inputs[2]
            self.action_cov = self.model.inputs[3]
            self.action_mean = self.model.get_layer("action_mean_l").output
        #self.trainable_params = self.model.trainable_weights
        #print(self.trainable_params)
        #self.action_mean = self.model(self.state)
        #self.action_dist = MultivariateNormalFullCovariance(self.action_mean, self.action_cov)
        #self.action_probability = self.action_dist.prob(self.action_taken)
        #self.action_sample =self.action_dist.sample()
        # Compile policy model with reinforcementLoss function

        # Get the action_mean("predicted" action) from the model, since the model output now is loss
        # provide K.learning_phase() because BatchNorm layers behave differently in training than in testing
        self.action_mean_func = K.function([self.state, K.learning_phase()], [self.action_mean])

        #params_updates = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, var_list=self.trainable_params)
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        #self.session = tf.Session()
        self.model_file = model_file

        #for param in self.trainable_params:
        #    #print(param.initializer)
        #    self.session.run(param.initializer)
        #    self.session.run(tf.global_variables_initializer())

    #def __enter__(self):
    #    return self

    #def __exit__(self, exception_type, exception_value, traceback):
    #    self.destroy()

    #def destroy(self):
    #    self.session.close()

    def sampleAction(self, state, action_cov):
        """
            NB: Only sample one step
        """
        #action_mean = self.model.predict(state.reshape(1, -1))
        action_mean = self.action_mean_func([state.reshape(1, -1), 0])[0]
        return np.random.multivariate_normal(action_mean.flatten(), action_cov)
        #return self.session.run(self.action_sample, feed_dict={self.state:state, K.learning_phase():0})

    def train(self, state, action_taken, advantage, action_cov, log_file):
        #for train_step in range(self.epochs):
        #    np.random.seed(self.seed)
        #    np.random.shuffle(action_taken)
        #    np.random.seed(self.seed)
        #    np.random.shuffle(state)
        #    np.random.seed(self.seed)
        #    np.random.shuffle(advantage)
        #    self.session.run(self.params_updates, feed_dict={self.action_taken:action_taken, self.state:state, self.advantage:advantage
        #                                                    , K.learning_phase():1})
        #    print("== Epoch: %d =="%(train_step))
        #    #if train_step % 10 == 0:
        #    loglike_vec, loss, action_prob = self.session.run([self.loglike, self.loss, self.action_probability], feed_dict={self.action_taken:action_taken, self.state:state, self.advantage:advantage
        #                                                                            , K.learning_phase():0})
        #    print("Log likelihood")
        #    print(loglike_vec)
        #    print("Loss")
        #    print(loss)
        #    #print("Action taken")
        #    #print(action_taken)
        #    print("Action prob")
        #    print(action_prob)
        #    print("Advantage")
        #    #print(advantage)
        inputs = {"state": state,
                "action_taken": action_taken,
                "advantage": advantage,
                "action_cov": action_cov}
        # Dummy output for the dummy loss function
        output = {"reinforcement_loss": np.zeros_like(advantage)}

        #K.set_learning_phase(True)
        self.model.fit(inputs, output, batch_size=self.batch_size, epochs=self.epochs,
                callbacks=[keras.callbacks.ModelCheckpoint(self.model_file, monitor='loss', verbose=1,
                    save_best_only=True, save_weights_only=False, mode='auto', period=self.epochs)
                    , keras.callbacks.CSVLogger(log_file, append=True)
                    , keras.callbacks.TerminateOnNaN()])

    def save(self, model_file):
        """
        Save the internal model to location model_file
        """
        self.model.save(model_file)
