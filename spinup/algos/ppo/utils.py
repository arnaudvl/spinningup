import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

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

def preprocess_obs(obs,grayscale=False,minmax=None,obs_old=None,frames=1,crop_size=None):
    """ Preprocess an observation from the gym environment.
    Eg: grayscale=True, minmax=(0,1), frames=3, crop_size=(50,50)
    
    Returns preprocessed observation.
    """
    if crop_size is not None: # crop image
        x, y, _ = obs.shape
        x_start = x//2-(crop_size[0]//2)
        y_start = y//2-(crop_size[1]//2) 
        obs = obs[x_start:x_start+crop_size[0],y_start:y_start+crop_size[1],:]      
    
    if grayscale:
        obs = np.dot(obs[...,:3], [0.299, 0.587, 0.114])
        obs = obs[...,np.newaxis] # add dimension
    
    if minmax is not None:
        min, max = minmax[0], minmax[1]
        obs_min, obs_max = obs.min(), obs.max()
        obs = ((obs - obs_min) / (obs_max - obs_min)) * (max - min) + min
    
    if obs_old is not None and frames>1: # keep last frames
        obs = np.concatenate((obs,obs_old[:,:,:-1]),axis=-1)
    
    if obs.shape[-1]<frames: # make sure output obs has right nb of channels
        pad = np.zeros(obs.shape[:2] + (frames-obs.shape[-1],))
        obs = np.concatenate((obs,pad),axis=-1)
    
    return obs