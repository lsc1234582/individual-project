import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import BatchNormalization, Dense, Input
from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

class GaussianPolicy:
    def modelLoss(advantages, action_cov):
        """
            Return a a Refinforce loss function to be optimised
        """
        def loss(action_pred, action_taken):
            loglike = -0.5 * tf.matmul(tf.square(action_taken - action_pred), tf.matrix_inverse(action_cov))
            return -tf.reduce_mean(loglike * advantages)
        return loss
    """

    self.action(6,) is sampled from self.policy,
    which is a gaussian distribution with mean as the output from self.model
    self.model takes state (24,) and produces an action mean (6,)
    TODO: Constrain self.model output?
    """
    def __init__(self, model_file, cov=None, lr=0.05, epochs=2, batch_size=128, seed=0, is_load_model=False):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, 24))
        self.action_taken = tf.placeholder(dtype=tf.float32, shape=(None, 6))
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.action_cov = np.eye(6).astype("float32") / 10.0 if cov==None else cov
        if not is_load_model:
            self.model = Sequential([
                #Input(shape=(24,), dtype="float32", name="state_l"),
                Dense(256, input_shape=(24,), dtype="float32", kernel_initializer="random_normal", kernel_regularizer=keras.regularizers.l2(0.01), bias_initializer="zeros", activation="relu", name="hidden_l1"),
                BatchNormalization(),
                Dense(64, kernel_initializer="random_normal",  bias_initializer="zeros", kernel_regularizer=keras.regularizers.l2(0.01), activation="relu", name="hidden_l2"),
                BatchNormalization(),
                Dense(6, kernel_initializer="random_normal", kernel_regularizer=keras.regularizers.l2(0.01), bias_initializer="zeros", name="action_l")
            ])
        else:
            # load an existing model
            self.model = load_model(model_file, custom_objects={'loss': GaussianPolicy.modelLoss(self.advantage,
                self.action_cov)})
        self.model.compile(loss=GaussianPolicy.modelLoss(self.advantage, self.action_cov), optimizer='rmsprop')
        #self.trainable_params = self.model.trainable_weights
        #print(self.trainable_params)
        self.action_mean = self.model(self.state)
        #self.action_dist = MultivariateNormalFullCovariance(self.action_mean, self.action_cov)
        #self.action_probability = self.action_dist.prob(self.action_taken)
        #self.action_sample =self.action_dist.sample()
        # Compile policy model with modelLoss function

        #params_updates = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, var_list=self.trainable_params)
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.session = tf.Session()
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

    def sampleAction(self, state):
        """
            NB: Only sample one step
        """
        action_mean = self.model.predict(state.reshape(1, -1))
        return np.random.multivariate_normal(action_mean.flatten(), self.action_cov)
        #return self.session.run(self.action_sample, feed_dict={self.state:state, K.learning_phase():0})

    def train(self, state, action_taken, advantage, log_file):
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
        self.model.compile(loss=GaussianPolicy.modelLoss(advantage, self.action_cov), optimizer='rmsprop')
        self.model.fit(state, action_taken, batch_size=action_taken.shape[0], shuffle=False, epochs=self.epochs,
                callbacks=[keras.callbacks.ModelCheckpoint(self.model_file, monitor='loss', verbose=1,
                    save_best_only=True, save_weights_only=False, mode='auto', period=self.epochs)
                    , keras.callbacks.CSVLogger(log_file, append=True)
                    , keras.callbacks.TerminateOnNaN()])

    def save(self):
        """
        Save the internal model to location self.model_file
        """
        self.model.save(self.model_file)
