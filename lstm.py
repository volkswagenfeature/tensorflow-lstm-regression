# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers


def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """


    def lstm_cells(layers):
        ## Is this fault checking? Should it raise a TypeError if layers[0] is not a dict?
        ## Again, if not exposed/not used by user, this shouldn't be needed. 
        if isinstance(layers[0], dict):
            ##tf.contrib.rnn.DropoutWrapper()
            ##Changes in 1.4 are entirely additions.
            ##original properties are:
            ## -output_size: Output tensor size??
            ## -state_size: Internal state?? Where? how? why is this accessable??
            ## - __init__ Unchanged, except the addition of a dropout_state_filter_visitor variable
            ## -- only affects state_keep_prob, isn't relevant here.
            ## - __call__ Completely identical
            ## - zero_state Completely identical

            ##tf.contrib.rnn.BasicLSTMCell()
            ##1.4 adds alias in tf.nn library. Maybe try to minimize use of contrib library?
            ##Something something adds stability??
            ##In fact, they recomend this. "For advanced models, please use the full tf.nn cell that follows 
            ##Original Properties: output_size, state_size. 1.4 has only additions
            ##Methods:
            ##__init__() 1.4 removes a depreciated input_size variable, which does nothing.
            ## --it also changes the format of the activation function, though functionality is the same.
            ## --state_is_tuple stays the same.
            ## zero_state() is identical in paramters and return values. 

            # For loop variant, for learning purpouses.
            rolling_result = list()
            for layer in layers:
                if (layer.get('keep_prob')):
                    rolling_result.append(tf.nn.rnn_cell.DropoutWrapper( 
                                            tf.nn.BasicLSTMCell(
                                                layer['num_units'], 
                                                state_is_tuple=True), 
                                            layer['keep_prob']))
                else:
                    rolling_result.append(tf.nn.BasicLSTMCell(
                                            layer['num_units'], 
                                            state_is_tuple=True))
            return rolling_result

            # Original list comprehension
#            return [tf.contrib.rnn.DropoutWrapper( tf.contrib.rnn.BasicLSTMCell( layer['num_units'], state_is_tuple=True),
#                layer['keep_prob']
#            )
#            if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(
#                    layer['num_units'],
#                    state_is_tuple=True
#                ) for layer in layers
#            ]

        ## This is the else clause of the previous if statement. 
        ## Very smilar to else clause at return above.
        ## only difference is that it's appending the full "layer", rather than layer['num_units']
        return [tf.nn.BasicLSTMCell(layer, state_is_tuple=True) for layer in layers]

    def dnn_layers(input_layers, layers):
        ## Why suport different formats?? The use of a variable that could be either
        ## a scalar type (int/char/whatever) or a dict (much more complex), or NoneType
        ## is interesting to me. Need to pay attention to use here.
        ## tflayers is alias for tf.contrib.layers
        if layers and isinstance(layers, dict):
            ## tflayers.stack() (alias of tf.contrib.layers.stack)
            ## Calls stack_layers repeatedly. What does stack_layers do?
            ## identical between versions.
            ## activation, dropout are kwargs passed directly to stack_layers
            ## In this layer, every cell is connected to every other cell.
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            ## Find out what activation and ropout parameters do, and why they're excluded here
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            ## Why does this exist? Should there be an exception here? in this case, the function
            ## does nothing.
            return input_layers

    # As used in lstm_sin_cos, X is features, y is labels.
    def _lstm_model(X, y):

        ## tf.contrib.rnn.MultiRNNCell (alias of tf.nn.rnn_cell.MultiRNNCell)
        ## Properties are unchanged (only additions)
        ## Methods:
        ## - __init__() unchanged. State_is_tuple=False is depreciated, but not used here.
        ## - __call__() unchanged.
        ## - zero_state() unchanged.
        stacked_lstm = tf.nn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        ## tf.unstack() 
        ## Functionality is unchanged, but the oposite operation has been changed from pack() to stack()
        ## Also, the numpy equivelent has changed from list(x) to np.unstack(x)
        x_ = tf.unstack(X, axis=1, num=num_units)
        ## tf.contrib.rnn.static_rnn (moved to tf.nn.static_rnn)
        ## Identical, except for the fact that it's been moved
        output, layers = tf.nn.static_rnn(stacked_lstm, x_, dtype=tf.float32)
        output = dnn_layers(output[-1], dense_layers)
        ## tflearn.models.linear_regression()
        ## No change. Corrected a typo in a comment between versions, but nothing else. 
        prediction, loss = tflearn.models.linear_regression(output, y)
        ## tf.contrib.layers.optimize_loss()
        ## now raises a ValueError if gradients is empty, look out for try/except statements.
        train_op = tf.contrib.layers.optimize_loss(
            ## tf.contrib.framework.get_global_step()
            ## Depreciated. New version is documented, old version is not. update, move it,
            ## and hope functionality is the same.
            loss, tf.train.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model
