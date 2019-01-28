import numpy as np
from keras.utils import Sequence


class TimeseriesGenerator(Sequence):
    """Utility class for generating batches of temporal data.

    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.

    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i]`, `data[i-r]`, ... `data[i - length]`
            are used for create a sample sequence.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `true`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).

    # Returns
        A [Sequence](/utils/#sequence) instance.

    # Examples

    ```python
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np

    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2)
    assert len(data_gen) == 20

    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                          np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y,
                          np.array([[10], [11]]))
    ```
    """

    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128):
        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __len__(self):
        return (self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride)

    def _empty_batch(self, num_rows):
        samples_shape = [num_rows, self.length // self.sampling_rate]
        samples_shape.extend(self.data.shape[1:])
        targets_shape = [num_rows]
        targets_shape.extend(self.targets.shape[1:])
        return np.empty(samples_shape), np.empty(targets_shape)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples, targets = self._empty_batch(len(rows))
        for j, row in enumerate(rows):
            indices = range(rows[j] - self.length, rows[j], self.sampling_rate)
            samples[j] = self.data[indices]
            targets[j] = self.targets[rows[j]]
        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets
