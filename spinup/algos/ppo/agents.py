import tensorflow as tf

from spinup.algos.ppo.custom_tf import CausalConv1D

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    
    print('\nMLP policy dimensions:')
    print('Input: {}'.format(x.get_shape()))
    x = tf.layers.flatten(x,name='flatten')
    print('Input flattened: {}'.format(x.get_shape()))
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
    
    print('\nCNN policy dimensions:')
    print('Input: {}'.format(x.get_shape()))
    
    for i in range(len(filters)):
        x = tf.layers.conv2d(x, filters[i], kernel_size[i], strides=strides[i],
                             padding=padding, activation=activation, name='conv2d_'+str(i))
        print('Conv2d: {}'.format(x.get_shape()))
    
    if flatten_type=='flatten':
        x = tf.layers.flatten(x, name=flatten_type)
    elif flatten_type=='global_average_pooling':
        x = tf.reduce_mean(x, axis=[1,2], name=flatten_type)
    elif flatten_type=='global_max_pooling':
        x = tf.reduce_max(x, axis=[1,2], name=flatten_type)
    
    print('Flatten: {}'.format(x.get_shape()))
    
    for n,h in enumerate(hidden_sizes[:-1]):
        x = tf.layers.dense(x, units=h, activation=activation, name='fc_'+str(n))
        print('Fully connected: {}'.format(x.get_shape()))
        
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