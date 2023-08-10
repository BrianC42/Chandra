'''
Created on Feb 19, 2021

Copied from https://www.tensorflow.org/tutorials/structured_data/time_series

@author: Brian
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class WindowGenerator():
    '''
    classdocs
    '''

    def __init__(self, input_width, label_width, shift, \
               df_train=None, df_values=None, df_test=None, \
               label_columns=None):
        '''
        Constructor
        '''
        # Store the raw data.
        self.df_train = df_train
        self.df_values = df_values
        self.df_test = df_test

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df_train.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice,:]
        labels = features[:, self.labels_slice,:]
        if self.label_columns is not None:
            labels = tf.stack([labels[:,:, self.column_indices[name]] for name in self.label_columns], \
                              axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # WindowGenerator.split_window = split_window
    
    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n,:, plot_col_index], \
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n,:, label_col_index], \
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n,:, label_col_index], \
                            marker='X', edgecolors='k', label='Predictions', \
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    # WindowGenerator.plot = plot

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float64)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, \
                                                                  targets=None, \
                                                                  sequence_length=self.total_window_size, \
                                                                  sequence_stride=1, \
                                                                  shuffle=True, \
                                                                  batch_size=32,)
        ds = ds.map(self.split_window)

        return ds

    # WindowGenerator.make_dataset = make_dataset
    
    @property
    def train(self):
        return self.make_dataset(self.df_train)

    @property
    def values(self):
        return self.make_dataset(self.df_values)

    @property
    def test(self):
        return self.make_dataset(self.df_test)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    # WindowGenerator.train = train
    # WindowGenerator.val = val
    # WindowGenerator.test = test
    # WindowGenerator.example = example
