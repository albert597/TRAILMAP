from models.model import input_dim, output_dim
from utilities.utilities import *
import random


# Load the data into
def load_data(data_path, nb_examples=None):
    volumes_folder_path = data_path + "/volumes"
    labels_folder_path = data_path + "/labels"

    volumes_path = get_dir(volumes_folder_path)
    labels_path = get_dir(labels_folder_path)

    assert len(labels_path) == len(volumes_path)

    total_length = len(labels_path)

    if nb_examples is None:
        nb_examples = total_length
    else:
        assert nb_examples <= total_length

    inds = list(range(len(volumes_path)))
    random.shuffle(inds)

    x = []
    y = []

    for i in range(nb_examples):
        rand_ind = inds[i]

        x.append(read_tiff_stack(volumes_path[rand_ind]))
        y.append(read_tiff_stack(labels_path[rand_ind]))

    inds = list(range(nb_examples))
    random.shuffle(inds)

    x_train = []
    y_train = []

    for i in inds:

        offset = (input_dim - output_dim)//2

        background = np.copy(crop_numpy(offset, offset, offset, y[i]))
        background[background == 2] = 0
        background[background == 3] = 0
        background[background == 4] = 0

        axons = np.copy(crop_numpy(offset, offset, offset, y[i]))
        axons[axons == 1] = 0
        axons[axons == 2] = 1
        axons[axons == 3] = 0
        axons[axons == 4] = 0

        artifact = np.copy(crop_numpy(offset, offset, offset, y[i]))
        artifact[artifact == 1] = 0
        artifact[artifact == 2] = 0
        artifact[artifact == 3] = 1
        artifact[artifact == 4] = 0

        edges = np.copy(crop_numpy(offset, offset, offset, y[i]))
        edges[edges == 1] = 0
        edges[edges == 2] = 0
        edges[edges == 3] = 0
        edges[edges == 4] = 1

        # 0 channel is segmentation
        # 1 channel is background
        # 2 channel is artifact
        # 3 channel is edge
        output = np.stack([axons, background, artifact, edges], axis=-1)

        input = x[i].reshape(x[i].shape + (1,))

        if np.count_nonzero(output == 1) > 0:
            x_train.append(input)
            y_train.append(output)

    x = np.array(x_train)
    y = np.array(y_train)

    return x, y
