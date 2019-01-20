import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

from spinup.algos.ppo.custom_tf import CausalConv1D

EPS = 1e-8

"""
Utils
"""

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

"""
Agents
"""

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    print('Pre flatten mlp: {}'.format(x.get_shape()))
    x = tf.layers.flatten(x,name='flatten')
    print('Flatten mlp: {}'.format(x.get_shape()))
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
        print('Hidden_size: {}'.format(x.get_shape()))
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='logits')


def lstm(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for n,h in enumerate(hidden_sizes[:-1]):
        seq=False if n+1==len(hidden_sizes[:-1]) else True # return sequences for intermediate LSTM layers
        x = tf.keras.layers.LSTM(h, activation=activation, return_sequences=seq, name='lstm_'+str(n))(x)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='logits')


def tcn(x, hidden_sizes=(32,), nb_filters=2, kernel_size=2, dilation_rate=2, dilation_depth=6, 
        use_bias=False, activation=tf.tanh, output_activation=None):
    
    n_features = inputs.get_shape().as_list()[2]
    x = CausalConv1D(n_features, 1, activation=activation, use_bias=use_bias, name='feature_selection_layer')(x)
    
    for i in range(0,dilation_depth+1):
        x = CausalConv1D(nb_filters, kernel_size, dilation_rate=dilation_rate**i, use_bias=use_bias,
                         activation=activation, name='dilated_conv1d_'+str(kernel_size**i))(x)
    
    x = tf.layers.flatten(x,name='flatten')
    
    for n,h in enumerate(hidden_sizes[:-1]):
        x = tf.layers.dense(x, units=h, activation=activation, name='fc_'+str(n))
        
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='logits')    
    

def cnn(x, hidden_sizes=(32,), filters=(16,32,64), kernel_size=(3,3,3), strides=(2,2,2), 
        padding='valid', flatten_type='flatten', activation=tf.tanh, output_activation=None):
    
    for i in range(len(filters)):
        print('Conv2d: {}'.format(x.get_shape()))
        x = tf.layers.conv2d(x, filters[i], kernel_size[i], strides=strides[i],
                             padding=padding, activation=activation, name='conv2d_'+str(i))
    
    print('Conv2d: {}'.format(x.get_shape()))
    
    if flatten_type=='flatten':
        x = tf.layers.flatten(x, name=flatten_type)
    elif flatten_type=='global_average_pooling':
        x = tf.reduce_mean(x, axis=[1,2], name=flatten_type)
    elif flatten_type=='global_max_pooling':
        x = tf.reduce_max(x, axis=[1,2], name=flatten_type)
    
    print('flatten: {}'.format(x.get_shape()))
    
    for n,h in enumerate(hidden_sizes[:-1]):
        x = tf.layers.dense(x, units=h, activation=activation, name='fc_'+str(n))
        print('fc: {}'.format(x.get_shape()))
        
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='logits')


def cnn_lstm(x, hidden_sizes=(32,), filters=(16,32,64), kernel_size=(3,3,3), strides=(2,2,2), 
        padding='valid', activation=tf.tanh, output_activation=None):
    
    for i in range(len(filters)):
        x = tf.layers.conv2d(x, filters[i], kernel_size[i], strides=strides[i],
                             padding=padding, activation=activation, name='conv2d_'+str(i))
    
    for n,h in enumerate(hidden_sizes[:-1]):
        seq=False if n+1==len(hidden_sizes[:-1]) else True # return sequences for intermediate LSTM layers
        x = tf.keras.layers.LSTM(h, activation=activation, return_sequences=seq, name='lstm_'+str(n))(x)
        
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='logits')


"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    
    print('Logits: {}'.format(logits.get_shape()))
    
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def lstm_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = lstm(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def lstm_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = lstm(x, list(hidden_sizes)+[act_dim], activation, None)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def cnn_categorical_policy(x, a, hidden_sizes, filters, kernel_size, strides, padding, 
                           flatten_type, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = cnn(x, hidden_sizes=list(hidden_sizes)+[act_dim], filters=filters, 
                 kernel_size=kernel_size, strides=strides, padding=padding, 
                 flatten_type=flatten_type, activation=activation, output_activation=None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def cnn_gaussian_policy(x, a, hidden_sizes, filters, kernel_size, strides, padding, 
                        flatten_type, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = cnn(x, hidden_sizes=list(hidden_sizes)+[act_dim], filters=filters, 
             kernel_size=kernel_size, strides=strides, padding=padding, 
             flatten_type=flatten_type, activation=activation, output_activation=None)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


"""
Actor-Critics
"""
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