"""
Andrew Player
September 2022
Script for training a network for MSTAR SAR Target Detection.
"""

import os
from math import ceil
import numpy as np
from mstar_io import load_sample
from model import create_net
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


class DataGenerator(Sequence):

    """
    Dataset Generator for sequencially passing files from storange into the model.
    """

    def __init__(self, file_list, data_path, tile_size):
        
        self.file_list = file_list
        self.tile_size = tile_size
        self.data_path = data_path
        self.on_epoch_end()


    def __len__(self):
        
        """
        The amount of files in the dataset.
        """

        return int(len(self.file_list))


    def __getitem__(self, index):
        
        """
        Returns the set of inputs and their corresponding truths.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index:(index + 1)]

        # single file
        file_list_temp = [self.file_list[k] for k in indexes]

        # Set of X_train and y_train
        X, y = self.__data_generation(file_list_temp)

        return X, y


    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.file_list))


    def __data_generation(self, file_list_temp):
        
        """
        Returns individual pairs of inputs and their corresponding truths.
        """

        # Generate data
        for ID in file_list_temp:

            magnitude, label = load_sample(os.path.join(self.data_path, ID))

            x = magnitude.reshape((1, self.tile_size, self.tile_size, 1))
            y = label.reshape((1, 7))

        return x, y


def train_model(model_name, dataset_dir, tile_size, num_epochs, batch_size):

    train_path = dataset_dir + '/train'
    val_path   = dataset_dir + '/validation'

    all_training_files   = os.listdir(train_path)
    all_validation_files = os.listdir(val_path)

    training_partition   = [item for item in all_training_files]
    validation_partition = [item for item in all_validation_files]

    training_generator = DataGenerator(training_partition, train_path, tile_size)
    validation_generator = DataGenerator(validation_partition, val_path, tile_size)

    model = create_net(
        model_name    = model_name,
        tile_size     = 128,
        label_count   = 7
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor  = 'loss',
        patience = 2,
        verbose  = 1
    )
    
    checkpoint = ModelCheckpoint(
        filepath       = 'models/checkpoints/' + model_name,
        monitor        = 'val_loss',
        mode           = 'min',
        verbose        = 1,
        save_best_only = True
    )

    training_samples   = len(training_partition)
    validation_samples = len(validation_partition)

    training_steps     = ceil(training_samples / batch_size)
    validation_steps   = ceil(validation_samples / batch_size)

    history = model.fit(
        training_generator,
        epochs           = num_epochs,
        validation_data  = validation_generator,
        batch_size       = batch_size,
        steps_per_epoch  = training_steps,
        validation_steps = validation_steps,
        callbacks        = [checkpoint, early_stopping]
    )

    model.save("models/" + model_name)
    
    return history