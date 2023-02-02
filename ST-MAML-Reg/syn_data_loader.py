
import torch
import numpy as np
from collections import namedtuple

Task = namedtuple('Task', ['x', 'y', 'task_info'])

def generate_sinusoid_batch(amp_range, phase_range, freq_range, input_range, num_samples,
                            batch_size, oracle):
    amp = np.random.uniform(amp_range[0], amp_range[1], [batch_size])
    phase = np.random.uniform(phase_range[0], phase_range[1], [batch_size])
    freq = np.random.uniform(freq_range[0], freq_range[1], [batch_size])
    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 2])
    for i in range(batch_size):
        dim_1 = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = amp[i] * np.sin(freq[i]*dim_1 - phase[i])
        dim_2 = np.ones_like(dim_1)
        inputs[i] = np.concatenate((dim_1, dim_2), axis=1)


    return inputs, outputs, amp, phase, freq


def generate_linear_batch(slope_range, intersect_range, input_range,
                          num_samples, batch_size, oracle):
    slope = np.random.uniform(slope_range[0], slope_range[1], [batch_size])
    intersect = np.random.uniform(intersect_range[0], intersect_range[1],
                                  [batch_size])
    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 2])
    for i in range(batch_size):

        dim_1 = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        dim_2 = np.ones_like(dim_1)
        inputs[i] = np.concatenate((dim_1, dim_2), axis=1)
        outputs[i] = dim_1 * slope[i] + intersect[i]

    return inputs, outputs, slope, intersect




def generate_quadratic_batch(quad_coef_range, linear_coef_range, constant_range, input_range,
                          num_samples, batch_size, oracle):

    quad_coef = np.random.uniform(quad_coef_range[0], quad_coef_range[1], [batch_size])
    linear_coef = np.random.uniform(linear_coef_range[0], linear_coef_range[1], [batch_size])
    constant_coef = np.random.uniform(constant_range[0], constant_range[1], [batch_size])


    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 2])
    for i in range(batch_size):
        dim_1 = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = quad_coef[i]*dim_1**2 + linear_coef[i]*dim_1 + constant_coef[i]
        dim_2 = np.ones_like(dim_1)
        inputs[i] = np.concatenate((dim_1, dim_2), axis=1)

    return inputs, outputs, quad_coef, linear_coef, constant_coef


def generate_cubic_batch(cubic_coef_range, quad_coef_range, linear_coef_range, constant_range, input_range,
                          num_samples, batch_size, oracle):

    cubic_coef = np.random.uniform(cubic_coef_range[0], cubic_coef_range[1], [batch_size])
    quad_coef = np.random.uniform(quad_coef_range[0], quad_coef_range[1], [batch_size])
    linear_coef = np.random.uniform(linear_coef_range[0], linear_coef_range[1], [batch_size])
    constant_coef = np.random.uniform(constant_range[0], constant_range[1], [batch_size])


    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 2])
    for i in range(batch_size):
        dim_1 = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = cubic_coef[i]*dim_1**3 + quad_coef[i]*dim_1**2 + linear_coef[i]*dim_1 + constant_coef[i]
        dim_2 = np.ones_like(dim_1)
        inputs[i] = np.concatenate((dim_1, dim_2), axis=1)

    return inputs, outputs, cubic_coef, quad_coef, linear_coef, constant_coef


def generate_quad_surf_batch(first_dim_coef_range, second_dim_coef_range, input_range,
                         num_samples, batch_size, oracle):

    first_dim_coef = np.random.uniform(first_dim_coef_range[0], first_dim_coef_range[1], [batch_size])
    second_dim_coef = np.random.uniform(second_dim_coef_range[0], second_dim_coef_range[1], [batch_size])

    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 2])
    for i in range(batch_size):
        dim_1 = np.random.uniform(input_range[0], input_range[1],
                                  [num_samples, 1])
        dim_2 = np.random.uniform(input_range[0], input_range[1], [num_samples, 1])
        outputs[i] = first_dim_coef[i]*dim_1**2 + second_dim_coef[i]*dim_2**2
        inputs[i] = np.concatenate((dim_1, dim_2), axis=1)

    return inputs, outputs, first_dim_coef, second_dim_coef


def generate_ripple_batch(input_coef_range, constant_range, input_range,
                         num_samples, batch_size, oracle):

    input_coef = np.random.uniform(input_coef_range[0], input_coef_range[1], [batch_size])
    constant_coef = np.random.uniform(constant_range[0], constant_range[1], [batch_size])

    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 2])
    for i in range(batch_size):
        dim_1 = np.random.uniform(input_range[0], input_range[1],
                                  [num_samples, 1])
        dim_2 = np.random.uniform(input_range[0], input_range[1], [num_samples, 1])
        outputs[i] = np.sin(-input_coef[i]*(dim_1**2 + dim_2**2)) + constant_coef[i]
        inputs[i] = np.concatenate((dim_1, dim_2), axis=1)

    return inputs, outputs, input_coef, constant_coef







class SimpleFunctionDataset(object):
    def __init__(self, num_total_batches=700, num_samples_per_function=15,
                 num_val_samples=10, meta_batch_size=75, oracle=False,
                 train=True, device='cpu', dtype=torch.float, **kwargs):
        self._num_total_batches = num_total_batches
        self._num_samples_per_function = num_samples_per_function
        self._num_val_samples = num_val_samples
        self._num_total_samples = num_samples_per_function
        self._meta_batch_size = meta_batch_size
        self._oracle = oracle
        self._train = train
        self._device = device
        self._dtype = dtype

    def _generate_batch(self):
        raise NotImplementedError('Subclass should implement _generate_batch')

    def __iter__(self):
        for batch in range(self._num_total_batches):
            inputs, outputs, infos = self._generate_batch()

            train_tasks = []
            val_tasks = []
            for task in range(self._meta_batch_size):
                task_inputs = torch.tensor(
                    inputs[task], device=self._device, dtype=self._dtype)
                task_outputs = torch.tensor(
                    outputs[task], device=self._device, dtype=self._dtype)
                task_infos = infos[task]
                train_task = Task(task_inputs[self._num_val_samples:],
                                  task_outputs[self._num_val_samples:],
                                  task_infos)
                train_tasks.append(train_task)
                val_task = Task(task_inputs[:self._num_val_samples],
                                task_outputs[:self._num_val_samples],
                                task_infos)
                val_tasks.append(val_task)
            yield train_tasks, val_tasks



class ManyFunctionsMetaDataset(SimpleFunctionDataset):
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0.0, 2*np.pi], freq_range=[0.8, 1.2],
                 input_range=[-5., 5.0],
                 slope_range=[-3.0, 3.0], intersect_range=[-3.0, 3.0],
                 quad_coef_range2=[-0.2, 0.2], linear_coef_range2=[-2., 2.], constant_range2=[-3., 3.],
                 cubic_coef_range3=[-0.1, 0.1], quad_coef_range3=[-0.2, 0.2], linear_coef_range3=[-2., 2.], constant_range3=[-3., 3.],
                 first_dim_coef_range=[-1, 1], second_dim_coef_range=[-1., 1.],
                 input_coef_range=[-0.2, 0.2], constant_range=[-3., 3.],
                 task_oracle=False,
                 noise_std=0, **kwargs):
        super(ManyFunctionsMetaDataset, self).__init__(**kwargs)
        # Task 1
        self._amp_range = amp_range
        self._phase_range = phase_range
        self._freq_range = freq_range

        # Task 2
        self._slope_range = slope_range
        self._intersect_range = intersect_range

        # Input
        self._input_range = input_range

        # Task 3
        self._quad_coef_range2 = quad_coef_range2
        self._linear_coef_range2 = linear_coef_range2
        self._constant_range2 = constant_range2

        # Task 4
        self._cubic_coef_range3 = cubic_coef_range3
        self._quad_coef_range3 = quad_coef_range3
        self._linear_coef_range3 = linear_coef_range3
        self._constant_range3 = constant_range3

        # Task 5
        self._first_dim_coef_range = first_dim_coef_range
        self._second_dim_coef_range = second_dim_coef_range

        # Task 6
        self._input_coef_range = input_coef_range
        self._constant_range = constant_range



        self._task_oracle = task_oracle
        self._noise_std = noise_std

        if not self._oracle:
            if not self._task_oracle:
                self.input_size = 2
            else:
                self.input_size = 2
        else:
            if not self._task_oracle:
                self.input_size = 3
            else:
                self.input_size = 4

        self.output_size = 1
        self.num_tasks = 6

    def _generate_batch(self):
        half_batch_size = self._meta_batch_size // 6
        sin_inputs, sin_outputs, amp, phase, freq = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            freq_range=self._freq_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        sin_task_infos = [{'task_id': 0, 'amp': amp[i], 'phase': phase[i], 'freq': freq[i]}
                          for i in range(len(amp))]



        lin_inputs, lin_outputs, slope, intersect = generate_linear_batch(
            slope_range=self._slope_range,
            intersect_range=self._intersect_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        lin_task_infos = [{'task_id': 1, 'slope': slope[i], 'intersect': intersect[i]}
                          for i in range(len(slope))]

        qua_inputs, qua_outputs, quad_coef2, linear_coef2, constant_coef2 = generate_quadratic_batch(
            quad_coef_range=self._quad_coef_range2,
            linear_coef_range=self._linear_coef_range2,
            constant_range=self._constant_range2,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)

        qua_task_infos = [{'task_id': 2, 'quad_coef2': quad_coef2[i], 'linear_coef2': linear_coef2[i], 'constant_coef2': constant_coef2[i]}
                      for i in range(len(quad_coef2))]



        cub_inputs, cub_outputs, cubic_coef3, quad_coef3, linear_coef3, constant_coef3 = generate_cubic_batch(
           cubic_coef_range=self._cubic_coef_range3,
           quad_coef_range=self._quad_coef_range3,
           linear_coef_range=self._linear_coef_range3,
           constant_range=self._constant_range3,
           input_range=self._input_range,
           num_samples=self._num_total_samples,
           batch_size=half_batch_size, oracle=self._oracle)
        cub_task_infos = [{'task_id': 3, 'cubic_coef3': cubic_coef3[i], 'quad_coef3': quad_coef3[i], 'linear_coef3': linear_coef3[i],
                           'constant_coef3': constant_coef3[i]} for i in range(len(quad_coef3))]

        quad_surf_inputs, quad_surf_outputs, quad_surf_first_dim_coef, quad_surf_second_dim_coef = generate_quad_surf_batch(
            first_dim_coef_range=self._first_dim_coef_range,
            second_dim_coef_range=self._second_dim_coef_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)

        quad_surf_task_infos = [{'task_id': 4, 'quad_surf_first_dim_coef': quad_surf_first_dim_coef[i],
                                'quad_surf_second_dim_coef': quad_surf_second_dim_coef[i]} for i in range(len(quad_surf_first_dim_coef))]

        ripple_inputs, ripple_outputs, ripple_input_coef, ripple_constant_coef = generate_ripple_batch(
            input_coef_range=self._input_coef_range,
            constant_range=self._constant_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)

        ripple_surf_task_infos = [{'task_id': 5, 'ripple_input_coef': ripple_input_coef[i],
                                'ripple_constant_coef': ripple_constant_coef[i]} for i in
                               range(len(ripple_input_coef))]






        inputs = np.concatenate((sin_inputs, lin_inputs, qua_inputs, cub_inputs, quad_surf_inputs, ripple_inputs))
        outputs = np.concatenate((sin_outputs, lin_outputs, qua_outputs, cub_outputs, quad_surf_outputs, ripple_outputs))

        if self._noise_std > 0:
            outputs = outputs + np.random.normal(scale=self._noise_std, size=outputs.shape)
        task_infos = sin_task_infos + lin_task_infos + qua_task_infos + cub_task_infos + quad_surf_task_infos + ripple_surf_task_infos
        return inputs, outputs, task_infos

