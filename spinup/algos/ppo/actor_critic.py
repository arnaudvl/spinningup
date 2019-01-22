import tensorflow as tf
from gym.spaces import Box, Discrete

from spinup.algos.ppo.agents import mlp, lstm, cnn
from spinup.algos.ppo.policies import mlp_gaussian_policy, mlp_categorical_policy, \
                                      lstm_gaussian_policy, lstm_categorical_policy, \
                                      cnn_gaussian_policy, cnn_categorical_policy

def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v

def lstm_actor_critic(x, a, hidden_sizes=(64,), activation=tf.tanh, 
                      output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = lstm_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = lstm_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(lstm(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v  
    
def cnn_actor_critic(x, a, hidden_sizes=(64,), filters=(16,32,64), kernel_size=(3,3,3), strides=(2,2,2), 
                     padding='valid', flatten_type='flatten', activation=tf.tanh, output_activation=None, 
                     policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = cnn_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = cnn_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, filters, kernel_size, strides, padding,
                                   flatten_type, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(cnn(x, list(hidden_sizes)+[1], filters, kernel_size, strides, 
                           padding, flatten_type, activation, None), axis=1)
    return pi, logp, logp_pi, v