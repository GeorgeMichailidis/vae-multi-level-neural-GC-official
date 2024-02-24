"""
utility functions for data manipulation
"""
import numpy as np
import math
import os


def parse_trajectories(trajs, num_val, num_test, context_length, prediction_length=0, stride=0, sample_freq=1, verbose=0):
    
    num_subjects = len(trajs)
    for subject_id, trajectory in trajs.items():

        input, target = seq_to_samples(x=trajectory,
                            context_length=context_length,
                            prediction_length=prediction_length,
                            stride=stride,
                            sample_freq=sample_freq,
                            verbose=verbose)

        input_target = np.concatenate([input, target],axis=1)

        ## allocate mem here; otherwise stack the list along axis=-1 later on will somehow get stuck ???
        if subject_id == 0:

            num_samples_total, length, num_nodes = input_target.shape
            num_train = int(num_samples_total-num_val-num_test)
            trajs_train = np.empty((num_train, num_subjects, length, num_nodes))
            trajs_val = np.empty((num_val, num_subjects, length, num_nodes))
            trajs_test = np.empty((num_test, num_subjects, length, num_nodes))

        trajs_train[:,subject_id,:,:] = input_target[:-(num_val + num_test)]
        trajs_val[:,subject_id,:,:] = input_target[-(num_val + num_test):-num_test]
        trajs_test[:,subject_id,:,:] = input_target[-num_test:]

    print(f'>> train.shape={trajs_train.shape}, val.shape={trajs_val.shape}, test.shape={trajs_test.shape}')
    return trajs_train, trajs_val, trajs_test


def seq_to_samples(x, context_length, prediction_length = 1, stride = 1, sample_freq=1, verbose=False):

    T, d = x.shape
    single_sample_length = (context_length + prediction_length - 1) * sample_freq + 1
    sample_size = math.floor((T - single_sample_length)/float(stride)) + 1

    input, target = [],[]
    for i in range(sample_size):

        input_start = int(i*stride)
        input_end = target_start = int(input_start + context_length * sample_freq)
        target_end = int(target_start + prediction_length * sample_freq)

        input.append(x[input_start:input_end][::sample_freq])
        target.append(x[target_start:target_end][::sample_freq])

    input, target = np.stack(input, axis=0), np.stack(target, axis=0)
    if verbose:
        print(f'x.shape={x.shape}; input.shape={input.shape}, target.shape={target.shape}')

    return (input, target)
