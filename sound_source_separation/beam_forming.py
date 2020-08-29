from typing import Iterable

import numpy as np


# todo: numpy 1.20.0 will support type annotation for ndarray
# https://numpy.org/devdocs/reference/typing.html
def calculate_steering_vector(mic_alignments, source_locations, frequencies,
                              sound_speed=340, use_far=False):
    """calculates steering vector

    Parameters
    ----------
    mic_alignments: 3 x M  dimensional ndarray [[x,y,z],[x,y,z]]
    source_locations: 3 x Ns dimensional ndarray [[x,y,z],[x,y,z] ]
    frequencies: Nk dimensional array (fourier transformed audio signal) [f1,f2,f3...]
    sound_speed: [m/s]
    use_far: if set to True, steering vector is calculated based on far assumption

    Returns steering vector (Nk x Ns x M)
    -------

    """
    n_channels = np.shape(mic_alignments)[1]

    if use_far:
        norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
        steering_phase = np.einsum('k,ism,ism->ksm', 2.j * np.pi / sound_speed * frequencies,
                                   norm_source_locations[..., None], mic_alignments[:, None, :])
        steering_vector = 1. / np.sqrt(n_channels) * np.exp(steering_phase)

    else:
        # distance: Ns x Nm
        distance = np.sqrt(np.sum(np.square(source_locations[..., None] - mic_alignments[:, None, :]), axis=0))
        delay = distance / sound_speed
        steering_phase = np.einsum('k,sm->ksm', -2.j * np.pi * frequencies, delay)
        steering_decay_ratio = 1. / distance
        steering_vector = steering_decay_ratio[None, ...] * np.exp(steering_phase)
        steering_vector = steering_vector / np.linalg.norm(steering_vector, 2, axis=2, keepdims=True)

    return steering_vector
