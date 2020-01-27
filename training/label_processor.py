import numpy as np

def process_labels(vol):

    for i in range(vol.shape[0]):
        slice = vol[i]

        if np.max(slice) > 0:
            for x in range(1, slice.shape[0] - 1):
                for y in range(1, slice.shape[1] - 1):
                    if slice[x][y] == 1 and is_axon_close(slice, x, y):
                        # Set edge label
                        # Note: Edge label is set to 3 because the whole slice is incremented by 1 in the end
                        slice[x][y] = 3

        # Increment whole slice by 1 to have background be 1
        slice += 1

    return vol


def is_axon_close(slice, x, y):
    return slice[x][y+1] == 2 or slice[x+1][y+1] == 2 or slice[x+1][y] == 2 or slice[x+1][y-1] == 2 or slice[x][y-1] == 2 or \
           slice[x-1][y-1] == 2 or slice[x-1][y] == 2 or slice[x-1][y+1] == 2
