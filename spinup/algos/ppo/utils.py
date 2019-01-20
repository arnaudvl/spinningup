import numpy as np

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