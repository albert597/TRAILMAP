import numpy as np

# 1 should be axons and 2 should be artifacts
def process_labels(vol):

    for i in range(vol.shape[0]):
        slice = vol[i]

        if np.max(slice) > 0:
            # Increment whole slice by 1 to have background be 1
            slice += 1
            for x in range(1, slice.shape[0] - 1):
                for y in range(1, slice.shape[1] - 1):
                    if slice[x][y] == 1 and is_axon_close(slice, x, y):
                        # Set edge label
                        slice[x][y] = 4

    return vol


def is_axon_close(slice, x, y):
    return slice[x][y+1] == 2 or slice[x+1][y+1] == 2 or slice[x+1][y] == 2 or slice[x+1][y-1] == 2 or slice[x][y-1] == 2 or \
           slice[x-1][y-1] == 2 or slice[x-1][y] == 2 or slice[x-1][y+1] == 2
