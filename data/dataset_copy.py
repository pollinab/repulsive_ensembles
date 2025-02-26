from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.matlib as npm

class Dataset(ABC):
    """A general dataset template that can be used as a simple and consistent
    interface. Note, that this is an abstract class that should not be
    instantiated.

        The original implementation can be found `here <https://github.com/chrhenning/hypercl>`__.


    In order to write an interface for another dataset, you have to implement
    an inherited class. You must always call the constructor of this base class
    first when instantiating the implemented subclass.
    
    Note, the internals are stored in the private member _data, that is
    described in the constructor.

    Attributes:
        classification: Whether the dataset is a classification or regression
            dataset.
        sequence: Whether the dataset contains sequences (samples have temporal
            structure).
            In case of a sequence dataset, the temporal structure can be
            decoded via the shape attributes of in- and outputs.
            Note, that all samples are internally zero-padded to the same
            length.
        num_classes: The number of classes for a classification task (None
            otherwise).
        is_one_hot: Whether output labels are one-hot encoded for a
            classification task (None otherwise).
        in_shape: The original shape of an input sample. Note, that samples are
            encoded by this class as individual vectors (e.g., an MNIST sample
            is ancoded as 784 dimensional vector, but its original shape is:
            [28, 28, 1]).
            A sequential sample is encoded by concatenating all timeframes.
            Hence, the number of timesteps can be decoded by dividing a single 
            sample vector by np.prod(in_shape).
        out_shape: The original shape of an output sample (see in_shape).
        num_train_samples: The number of training samples.
        num_test_samples: The number of test samples.
        num_val_samples: The number of validation samples.
        shuffle_test_samples: Whether the method 'next_test_batch' returns
                              test samples in random order at every epoch.
                              Defaults to True, i.e., samples have a random
                              ordering every epoch.
        shuffle_val_samples: Same as 'shuffle_test_samples' for samples from the
                             validation set.
    """

    def __init__(self):
        """Initialization of class attributes. The special (and private) data
        dictionary is also defined here. It can be used to easily store the
        processed dataset locally in a python readible format. This might speed
        up the dataset loading on the next program run.
        """
        # Internally, everything is stored in a certain structure, such that it
        # can easily be backuped (for instance via pickle).
        data = {}

        # Boolean: See attribute "classification".
        data.setdefault('classification', None)
        # Boolean: See attribute "sequence".
        data.setdefault('sequence', None)

        # Integer: See attribute "num_classes".
        data.setdefault('num_classes', None)
        # Integer: See attribute "is_one_hot".
        data.setdefault('is_one_hot', None)

        # A 2D numpy array, containing a sample input in each row (all samples
        # are encoded as single vectors.)
        data.setdefault('in_data', None)
        # A 2D numpy array, containing a sample output in each row (all samples
        # are encoded as single vectors.)
        data.setdefault('out_data', None)

        # List or numpy array: See attribute "in_shape".
        data.setdefault('in_shape', [])
        # List or numpy array: See attribute "in_shape".
        data.setdefault('out_shape', [])

        # List or numpy array: All row indices of "in_data" or "out_data", that
        # correspond to samples belonging to the training set.
        data.setdefault('train_inds', [])
        # List or numpy array: All row indices of "in_data" or "out_data", that
        # correspond to samples belonging to the test set.
        data.setdefault('test_inds', [])
        # List or numpy array: All row indices of "in_data" or "out_data", that
        # correspond to samples belonging to the validation set.
        data.setdefault('val_inds', None)

        self._data = data

        # These are other private attributes, that are not in the data dict
        # as there would be no reason to pickle them.
        self._batch_gen_train = None
        self._batch_gen_test = None
        self._batch_gen_val = None

        # We only need to fit the one-hot encoder for this dataset once.
        self._one_hot_encoder = None

        self._shuffle_test_samples = True
        self._shuffle_val_samples = True

    # TODO deprecate this attribute. Instead, distinguish between multi and
    # single label encoding.
    @property
    def classification(self):
        """Getter for read-only attribute classification."""
        return self._data['classification']

    @property
    def sequence(self):
        """Getter for read-only attribute sequence."""
        return self._data['sequence']

    @property
    def num_classes(self):
        """Getter for read-only attribute num_classes."""
        return self._data['num_classes']

    @property
    def is_one_hot(self):
        """Getter for read-only attribute is_one_hot."""
        return self._data['is_one_hot']

    @property
    def in_shape(self):
        """Getter for read-only attribute in_shape"""
        return self._data['in_shape']

    @property
    def out_shape(self):
        """Getter for read-only attribute out_shape"""
        return self._data['out_shape']

    @property
    def num_train_samples(self):
        """Getter for read-only attribute num_train_samples"""
        return np.size(self._data['train_inds'])

    @property
    def num_test_samples(self):
        """Getter for read-only attribute num_test_samples"""
        return np.size(self._data['test_inds'])

    @property
    def num_val_samples(self):
        """Getter for read-only attribute num_val_samples"""
        if self._data['val_inds'] is None:
            return 0
        return np.size(self._data['val_inds'])

    @property
    def shuffle_test_samples(self):
        """Getter for read-only attribute shuffle_test_samples"""
        return self._shuffle_test_samples

    @shuffle_test_samples.setter
    def shuffle_test_samples(self, value):
        """Setter for the attribute shuffle_test_samples.

        Note, a call to this method will reset the current generator, such that
        the next call to the method 'next_test_batch' results in starting a
        sweep through a new epoch (full batch).
        """
        self._shuffle_test_samples = value
        self._batch_gen_test = None

    @property
    def shuffle_val_samples(self):
        """Getter for read-only attribute shuffle_val_samples"""
        return self._shuffle_val_samples

    @shuffle_val_samples.setter
    def shuffle_val_samples(self, value):
        """Setter for the attribute shuffle_val_samples.

        See documentation of setter for attribute 'shuffle_test_samples'.
        """
        self._shuffle_val_samples = value
        self._batch_gen_val = None

    def get_train_inputs(self):
        """Get the inputs of all training samples.
        
        Note, that each sample is encoded as a single vector. One may use the
        attribute "in_shape" to decode the actual shape of an input sample.
        
        Returns:
            A 2D numpy array, where each row encodes a training sample.
        """
        return self._data['in_data'][self._data['train_inds'], :]

    def get_test_inputs(self):
        """Get the inputs of all test samples.
        
        See documentation of method "get_train_inputs" for details.
        
        Returns:
            A 2D numpy array.
        """
        return self._data['in_data'][self._data['test_inds'], :]

    def get_val_inputs(self):
        """Get the inputs of all validation samples.

        See documentation of method "get_train_inputs" for details.

        Returns:
            A 2D numpy array. Returns None if no validation set exists.
        """
        if self._data['val_inds'] is None:
            return None
        return self._data['in_data'][self._data['val_inds'], :]

    def get_train_outputs(self, use_one_hot=None):
        """Get the outputs (targets) of all training samples.

        Note, that each sample is encoded as a single vector. One may use the
        attribute "out_shape" to decode the actual shape of an output sample.
        Keep in mind, that classification samples might be one-hot encoded.

        Args:
            use_one_hot: For classification samples, the encoding of the
                returned samples can be either "one-hot" or "class index". This
                option is ignored for datasets other than classification sets.
                If None, the dataset its default encoding is returned.

        Returns:
            A 2D numpy array, where each row encodes a training target.
        """
        out_data = self._data['out_data'][self._data['train_inds'], :]
        return self._get_outputs(out_data, use_one_hot)

    def get_test_outputs(self, use_one_hot=None):
        """Get the outputs (targets) of all test samples.

        See documentation of method "get_train_outputs" for details.

        Args:
            use_one_hot: See "get_train_outputs".

        Returns:
            A 2D numpy array.
        """
        out_data = self._data['out_data'][self._data['test_inds'], :]
        return self._get_outputs(out_data, use_one_hot)
    
    def get_val_outputs(self, use_one_hot=None):
        """Get the outputs (targets) of all validation samples.

        See documentation of method "get_train_outputs" for details.

        Args:
            use_one_hot: See "get_train_outputs".

        Returns:
            A 2D numpy array. Returns None if no validation set exists.
        """
        if self._data['val_inds'] is None:
            return None
        out_data = self._data['out_data'][self._data['val_inds'], :]
        return self._get_outputs(out_data, use_one_hot)

    def next_train_batch(self, batch_size, use_one_hot=None):
        """Return the next random training batch.

        If the behavior of this method should be reproducible, please define a
        numpy random seed.

        Args:
            batch_size: The size of the returned batch.
            use_one_hot: See "get_train_outputs".

        Returns:
            [batch_inputs, batch_outputs] - A 2-element list, where each
            element is a 2D numpy array.
            batch_inputs: The inputs of the samples belonging to the batch.
            batch_outputs: The outputs of the samples belonging to the batch.
        """
        if self._batch_gen_train is None:
            self.reset_batch_generator(train=True, test=False, val=False)

        batch_inds = np.fromiter(self._batch_gen_train, np.int32,
                                 count=batch_size)
        return [self._data['in_data'][batch_inds, :],
                self._get_outputs(self._data['out_data'][batch_inds, :],
                                  use_one_hot)]

    def next_test_batch(self, batch_size, use_one_hot=None):
        """Return the next random test batch.

        See documentation of method "next_train_batch" for details.

        Args:
            batch_size: The batch size.
            use_one_hot: See "get_train_outputs".

        Returns:
            A 2-element list, where each element is a 2D numpy array,
            comprising either sample inputs or outputs.
        """
        if self._batch_gen_test is None:
            self.reset_batch_generator(train=False, test=True, val=False)

        batch_inds = np.fromiter(self._batch_gen_test, np.int32,
                                 count=batch_size)
        return [self._data['in_data'][batch_inds, :],
                self._get_outputs(self._data['out_data'][batch_inds, :],
                                  use_one_hot)]

    def next_val_batch(self, batch_size, use_one_hot=None):
        """Return the next random validation batch.

        See documentation of method "next_train_batch" for details.

        Args:
            batch_size: The batch size.
            use_one_hot: See "get_train_outputs".

        Returns:
            A 2-element list, where each element is a 2D numpy array,
            comprising either sample inputs or outputs. Returns None if no
            validation set exists.
        """
        if self._data['val_inds'] is None:
            return None

        if self._batch_gen_val is None:
            self.reset_batch_generator(train=False, test=False, val=True)

        batch_inds = np.fromiter(self._batch_gen_val, np.int32,
                                 count=batch_size)
        return [self._data['in_data'][batch_inds, :],
                self._get_outputs(self._data['out_data'][batch_inds, :],
                                  use_one_hot)]

    def reset_batch_generator(self, train=True, test=True, val=True):
        """The batch generation possesses a memory. Hence, the samples returned
        depend on how many samples already have been retrieved via the next-
        batch functions (e.g., "next_train_batch"). This method can be used to
        reset these generators.

        Args:
            train: If True, the generator for "next_train_batch" is reset.
            test: If True, the generator for "next_test_batch" is reset.
            val: If True, the generator for "next_val_batch" is reset, if a
                validation set exists.
        """
        if train:
            self._batch_gen_train = \
                Dataset._get_random_batch(self._data['train_inds'])

        if test:
            self._batch_gen_test = \
                Dataset._get_random_batch(self._data['test_inds'],
                                          self._shuffle_test_samples)

        if val and self._data['val_inds'] is not None:
            self._batch_gen_val = \
                Dataset._get_random_batch(self._data['val_inds'],
                                          self._shuffle_val_samples)

    @abstractmethod
    def get_identifier(self):
        """Returns the name of the dataset."""
        pass

    def is_image_dataset(self):
        """Are input (resp. output) samples images?
        
        Note, for sequence datasets, this method just returns whether a single
        frame encodes an image.
        
        Returns:
            A tuple (input_is_img, output_is_img) of booleans.
        """
        # Note, if these comparisons do not hold, the method has to be
        # overwritten.
        in_img = np.size(self.in_shape) == 3 and self.in_shape[-1] in [1, 3, 4]
        out_img = np.size(self.out_shape) == 3 and \
            self.out_shape[-1] in [1, 3, 4]
        return (in_img, out_img)

    def tf_input_map(self, mode='inference'):
        """This method should be used by the map function of the Tensorflow
        Dataset interface (tf.data.Dataset.map). In the default case, this is
        just an identity map, as the data is already in memory.

        There might be cases, in which the full dataset is too large for the
        working memory, and therefore the data currently needed by Tensorflow
        has to be loaded from disk. This function should be used as an
        interface for this process.

        Args:
            mode: This is the same as the mode attribute in the class
                  NetworkBase, that can be used to distinguish between training
                  and inference (e.g., if special input processing should be
                  used during training).

        Returns:
            A function handle, that maps the given input tensor to the
            preprocessed input tensor.
        """
        return lambda x : x

    def tf_output_map(self, mode='inference'):
        """Similar to method 'tf_input_map', just for dataset outputs.

        Note, in this default implementation, it is also just an identity map.
        """
        return lambda x : x

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, subclasses might overwrite this method and add data preprocessing/
        augmentation.

        Args:
            x: A 2D numpy array, containing inputs as provided by this dataset.
            device: The PyTorch device onto which the input should be mapped.
            mode: This is the same as the mode attribute in the class
                  NetworkBase, that can be used to distinguish between training
                  and inference (e.g., if special input processing should be
                  used during training).
                  Valid values are: 'train' and 'inference'.
            force_no_preprocessing: In case preprocessing is applied to the
                inputs (e.g., normalization or random flips/crops), this option
                can be used to prohibit any kind of manipulation. Hence, the
                inputs are transformed into PyTorch tensors on an "as is" basis.

        Returns:
            The given input x as PyTorch tensor.
        """
        # Note, this import is only needed for the functions:
        # input_to_torch_tensor() and output_to_torch_tensor()
        from  torch import from_numpy
        return from_numpy(x).float().to(device)

    def output_to_torch_tensor(self, y, device, mode='inference',
                               force_no_preprocessing=False):
        """Similar to method 'input_to_torch_tensor', just for dataset outputs.

        Note, in this default implementation, it is also does not perform any
        data preprocessing.
        """
        from  torch import from_numpy
        return from_numpy(y).float().to(device)

    def plot_samples(self, title, inputs, outputs=None, predictions=None,
                     num_samples_per_row=4, show=True, filename=None,
                     interactive=False, figsize=(10, 6)):
        """Plot samples belonging to this dataset. Each sample will be plotted
        in its own subplot.
        
        Args:
            title: The title of the whole figure.
            inputs: A 2D numpy array, where each row is an input sample.
            outputs (optional): A 2D numpy array of actual dataset targets.
            predictions (optional): A 2D numpy array of predicted output
                samples (i.e., output predicted by a neural network).
            num_samples_per_row (default: 4): Maximum number of samples plotted
                per row in the generated figure.
            show (default: True): Whether the plot should be shown.
            filename (optional): If provided, the figure will be stored under
                this filename.
            interactive (default: False): Turn on interactive mode. We mainly
                use this option to ensure that the program will run in
                background while figure is displayed. The figure will be
                displayed until another one is displayed, the user closes it or
                the program has terminated. If this option is deactivated, the
                program will freeze until the user closes the figure.
                Note, if using the iPython inline backend, this option has no
                effect.
            figsize (default: (10, 6)): A tuple, determining the size of the
                figure in inches.
        """
        # Determine the configs for the grid of this figure.
        pc = self._plot_config(inputs, outputs=outputs,
                               predictions=predictions)
        
        # Reverse one-hot encoding.
        if self.classification:
            num_time_steps = 1
            if self.sequence:
                num_time_steps = self._data['out_data'].shape[1] // \
                                 np.prod(self.out_shape)

            one_hot_size = num_time_steps * self.num_classes
            if outputs is not None and outputs.shape[1] == one_hot_size:
                outputs = self._to_one_hot(outputs, True)
                
            # Note, we don't reverse the encoding for predictions, as this
            # might be important for the subsequent plotting method.

        num_plots = inputs.shape[0]
        num_cols = int(min(num_plots, num_samples_per_row))
        num_rows = int(np.ceil(num_plots / num_samples_per_row))

        fig = plt.figure(figsize=figsize)
        outer_grid = gridspec.GridSpec(num_rows, num_cols,
                                       wspace=pc['outer_wspace'],
                                       hspace=pc['outer_hspace'])
        
        plt.suptitle(title, size=20)
        if interactive:
            plt.ion()
        
        outs = None
        preds = None
        
        for i in range(num_plots):
            inner_grid = gridspec.GridSpecFromSubplotSpec(pc['num_inner_rows'],
                pc['num_inner_cols'], subplot_spec=outer_grid[i],
                wspace=pc['inner_wspace'], hspace=pc['inner_hspace'])

            if outputs is not None:
                outs = outputs[i, :]
            if predictions is not None:
                preds = predictions[i, :]

            self._plot_sample(fig, inner_grid, pc['num_inner_plots'], i,
                              inputs[i, np.newaxis], outputs=outs, 
                              predictions=preds)

        if show:
            plt.show()

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

    @abstractmethod
    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Add a custom sample plot to the given Axes object.

        Note, this method is called by the "plot_samples" method.

        Note, that the number of inner subplots is configured via the method:
            _plot_config

        Args:
            fig: An instance of class matplotlib.figure.Figure, that will
                contains the given Axes object.
            inner_grid: An object of the class
                matplotlib.gridspec.GridSpecFromSubplotSpec. It can be used to
                access the subplots of a single sample via
                    ax = plt.Subplot(fig, inner_grid[i])
                where i is a number between 0 and num_inner_plots-1.
                The retrieved axes has to be added to the figure via:
                    fig.add_subplot(ax)
            num_inner_plots: The number inner subplots.
            ind: The index of the "outer" subplot.
            inputs: A 2D numpy array, containing a single sample (1 row).
            outputs (optional): A 2D numpy array, containing a single sample 
                (1 row). If this is a classification dataset, then samples are
                given as single labels (not one-hot encoded, irrespective of
                the attribute is_one_hot).
            predictions (optional): A 2D numpy array, containing a single 
                sample (1 row).
        """
        pass

    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Defines properties, used by the method 'plot_samples'.

        This method can be overwritten, if these configs need to be different
        for a certain dataset.

        Args:
            The given arguments, are the same as the same-named arguments of
            the method 'plot_samples'. They might be used by subclass
            implementations to determine the configs.

        Returns:
            A dictionary with the plot configs.
        """
        plot_configs = dict()
        plot_configs['outer_wspace'] = 0.4
        plot_configs['outer_hspace'] = 0.4
        plot_configs['inner_hspace'] = 0.2
        plot_configs['inner_wspace'] = 0.2
        plot_configs['num_inner_rows'] = 1
        plot_configs['num_inner_cols'] = 1
        plot_configs['num_inner_plots'] = 1

        return plot_configs

    def _get_outputs(self, data, use_one_hot=None):
        """A helper method for the output data getter methods. It will ensure,
        that the output encoding is correct.

        Args:
            data: The data to be returned (maybe after a change of encoding).
            use_one_hot: How data should be encoded.

        Returns:
            See documentation of method "get_train_outputs".
        """
        if self.classification:
            if use_one_hot is None:
                use_one_hot = self.is_one_hot

            if use_one_hot != self.is_one_hot:
                # Toggle current encoding.
                if self.is_one_hot:
                    return self._to_one_hot(data, reverse=True)
                else:
                    return self._to_one_hot(data)

        return data

    def _to_one_hot(self, labels, reverse=False):
        """ Transform a list of labels into a 1-hot encoding.

        Args:
            labels: A list of class labels.
            reverse: If true, then one-hot encoded samples are transformed back
                to categorical labels.

        Returns:
            The 1-hot encoded labels.
        """
        if not self.classification:
            raise RuntimeError('This method can only be called for ' +
                                   'classification datasets.')

        # Initialize encoder.
        if self._one_hot_encoder is None:
            self._one_hot_encoder = OneHotEncoder( \
                categories=[range(self.num_classes)])
            num_time_steps = 1
            if self.sequence:
                num_time_steps = labels.shape[1] // np.prod(self.out_shape)
            self._one_hot_encoder.fit(npm.repmat(
                    np.arange(self.num_classes), num_time_steps, 1).T)

        if reverse:
            # Unfortunately, there is no inverse function in the OneHotEncoder
            # class. Therefore, we take the one-hot-encoded "labels" samples
            # and take the indices of all 1 entries. Note, that these indices
            # are returned as tuples, where the second column contains the
            # original column indices. These column indices from "labels"
            # mudolo the number of classes results in the original labels.
            return np.reshape(np.argwhere(labels)[:,1] % self.num_classes, 
                              (labels.shape[0], -1))
        else:
            return self._one_hot_encoder.transform(labels).toarray()

    @staticmethod
    def _get_random_batch(indices, shuffle=True):
        """Returns a generator that yields sample indices from the randomly
        shuffled given dataset (prescribed by those indices). After a whole
        sweep through the array is completed, the indices are shuffled again.

        The implementation is inspired by another method I've implemented:
            The method "random_shuffle_loop": from https://git.io/fNDZJ

        Note, to get reproducable behavior, set a numpy random seed.

        Args:
            indices: A 1D numpy array of indices.
            shuffle (default: True): Whether the indices are shuffled every
                                     epoch. If this option is deactivated, the
                                     indices are simply processed in the
                                     given order every round.

        Returns:
            A generator object, that yields indices from the given array.

        Example Usage:
        >>> import iterlist
        >>> batch_size = 32
        >>> batch_generator = Dataset._get_random_batch(indices)
        >>> batch_inds = np.array( \
        ...     list(itertools.islice(batch_generator, batch_size)))
        
        Runtime Benchmark: What is a fast way to call the method?
        Note, that np.fromiter only works since indices is a 1D array.
        >>> import timeit
        >>> setup = '''
        ... import numpy as np
        ... import itertools
        ... from dataset import Dataset
        ... 
        ... indices = np.random.randint(0, high=100, size=100)
        ... 
        ... batch_size = 32
        ... batch_generator = Dataset._get_random_batch(indices)
        ... '''
        >>> print(timeit.timeit(
        ...     stmt="batch_inds = np.array(list(itertools.islice(" +
        ...          "batch_generator, batch_size)))",
        ...     setup=setup, number=100000))
        1.1637690339994151
        >>> print(timeit.timeit(
        ...     stmt="batch_inds = np.stack([next(batch_generator) " +
        ...          "for i in range(batch_size)])",
        ...     setup=setup, number=100000))
        6.16505571999005
        >>> print(timeit.timeit(
        ...     stmt="batch_inds = np.fromiter(itertools.islice(" +
        ...         batch_generator, batch_size), int)",
        ...     setup=setup, number=100000))
        0.9456974960048683
        >>> print(timeit.timeit(
        ...     stmt="batch_inds = np.fromiter(batch_generator, int, " +
        ...                                    count=batch_size)",
        ...     setup=setup, number=100000))
        0.871306327986531
        """
        num_samples = np.shape(indices)[0]
        arr_inds = np.arange(num_samples)
        i = num_samples
        while True:
            if i == num_samples:
                i = 0
                if shuffle:
                    np.random.shuffle(arr_inds)
            yield indices[arr_inds[i]]
            i += 1

if __name__ == '__main__':
    pass


