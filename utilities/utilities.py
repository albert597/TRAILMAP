import numpy as np
import cv2
import os
from os import listdir, makedirs
from os.path import join
from PIL import Image
import shutil
import sys

def crop_numpy(dim1, dim2, dim3, vol):
    return vol[dim1:vol.shape[0] - dim1, dim2:vol.shape[1] - dim2, dim3:vol.shape[2] - dim3]


def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)


def get_dir(path):
    tiffs = [join(path, f) for f in listdir(path) if f[0] != '.']
    return sorted(tiffs)


def crop_cube(x, y, z, vol, cube_length=64):
    # Cube shape
    return crop_box(x, y, z, vol, (cube_length, cube_length, cube_length))


def crop_box(x, y, z, vol, shape):
    return vol[z:z + shape[2], x:x + shape[0], y:y + shape[1]]


"""
Read images from start_index to end_index from a folder

@param path: The path to the folder
@param start_index: The index of the image to start reading from inclusive
@param end_index: The end of the image to stop reading from exclusive

@raise FileNotFoundError: If the path to the folder cannot be found 
"""
def read_folder_section(path, start_index, end_index):
    fnames = get_dir(path)
    vol = []

    for f in fnames[start_index: end_index]:
        img = cv2.imread(f, cv2.COLOR_BGR2GRAY)
        vol.append(img)

    vol = np.array(vol)

    return vol


def read_folder_stack(path):
    fnames = get_dir(path)

    fnames.sort()
    vol = cv2.imread(fnames[0], cv2.COLOR_BGR2GRAY)

    if len(vol.shape) == 3:
        return vol

    vol = []

    for f in fnames:
        img = cv2.imread(f, cv2.COLOR_BGR2GRAY)
        vol.append(img)

    vol = np.array(vol)

    return vol

def write_folder_stack(vol, path):

    if os.path.exists(path):
        print("Overwriting " + path)
        shutil.rmtree(path)

    makedirs(path)

    for i in range(vol.shape[0]):

        fname = os.path.join(path, "slice" + str(i).zfill(5) + ".tiff")
        cv2.imwrite(fname, vol[i])


def read_tiff_stack(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)

    return np.array(images)


def coordinate_vol(coords, shape):
    vol = np.zeros(shape, dtype="uint16")
    for c in coords:
        vol[c[0], c[1], c[2]] = 1
    return vol


def preprocess(vol):
    return vol / 65535


def preprocess_batch(batch):
    assert len(batch.shape) == 5
    lst = []

    for i in range(batch.shape[0]):
        lst.append(preprocess(batch[i]))

    return np.array(lst)


def dist(p1, p2):
    sqr = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    return sqr ** .5



"""
Progress bar to indicate status of the segment_brain function
"""

def draw_progress_bar(percent, eta="", bar_len = 40):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:>3.0f}%       {:20}".format("=" * int(bar_len * percent), bar_len, percent * 100, eta))
    sys.stdout.flush()
