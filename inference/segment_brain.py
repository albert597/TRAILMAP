from models.model import input_dim, output_dim
import numpy as np
import os
import time
import cv2
import sys
from PIL import Image

# Will need to be adjusted depending on the GPU
batch_size = 15

# Don't run network on chunks which don't have a value above threshold
threshold = 0.01

# Edge width between the output and input volumes
dim_offset = (input_dim - output_dim) // 2



"""
Progress bar to indicate status of the segment_brain function
"""

def draw_progress_bar(percent, eta, bar_len = 40):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:>3.0f}%       {:20}".format("=" * int(bar_len * percent), bar_len, percent * 100, eta))
    sys.stdout.flush()


def get_dir(path):
    tiffs = [os.path.join(path, f) for f in os.listdir(path) if f[0] != '.']

    return sorted(tiffs)

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
    for i in range(start_index, end_index):

        if i < 0:
            first_img = cv2.imread(fnames[0], cv2.COLOR_BGR2GRAY)
            vol.append(first_img)
        elif i >= len(fnames):
            last_img = cv2.imread(fnames[-1], cv2.COLOR_BGR2GRAY)
            vol.append(last_img)
        else:
            img = cv2.imread(fnames[i], cv2.COLOR_BGR2GRAY)
            vol.append(img)

    vol = np.array(vol)

    return vol



def write_folder_section(output_folder, file_names, section_index, section_seg):
    # Write the segmentation into the output_folder
    for slice_index in range(dim_offset, input_dim - dim_offset):
        input_file_name = file_names[section_index + slice_index]

        output_file_name = "seg-" + os.path.basename(input_file_name)
        output_full_path = output_folder + "/" + output_file_name

        # If write fails print problem
        pil_image = Image.fromarray(section_seg[slice_index])
        pil_image.save(output_full_path)
"""
Segment a brain by first cropping a list of chunks from the brain of the models's input size and executing the models on 
a batch of chunks. To conserve memory, the function will load sections of the brain at once.   

@param input_folder: The input directory that is a folder of 2D tiff files of the brain 
@param output_folder: Directory to write the segmentation files

@raise FileNotFoundError: If input_folder cannot be found
@raise NotADirectoryError: If the input_folder is not a directory
"""


def segment_brain(input_folder, output_folder, model):

    # Name of folder
    folder_name = os.path.basename(input_folder)
    # Get the list of tiff files
    file_names = get_dir(input_folder)

    if len(file_names) < 36:
        print("The Z direction must contain a minimum of 36 images")
        return

    first_img = cv2.imread(file_names[0], cv2.COLOR_BGR2GRAY)

    if first_img.shape[0] < 36 or first_img.shape[1] < 36:
        print("The X and Y direction must contain a minimum of 36 pixels")
        return


    eta = "ETA: Pending"
    # Get start time in minutes. Needed to calculate ETA
    start_time = time.time()/60

    total_sections = (len(file_names) // output_dim) * output_dim
    print("Name: " + folder_name)

    draw_progress_bar(0, eta)
    # Each iteration of loop will cut a section from slices i to i + input_dim and run helper_segment_section

    section_index = -dim_offset
    while section_index <= len(file_names) - input_dim + dim_offset:

        # Read section of folder
        section = read_folder_section(input_folder, section_index, section_index + input_dim).astype('float32')

        # Make the volume pixel intensity between 0 and 1
        section_vol = section / (2 ** 16 - 1)

        # Get the segmentation of this chunk
        section_seg = helper_segment_section(model, section_vol)

        # Write to output folder
        write_folder_section(output_folder, file_names, section_index, section_seg)

        # Calculate ETA
        now_time = time.time()/60
        sections_left = ((total_sections - section_index)/output_dim) - 1
        time_per_section = (now_time - start_time)/(1 + section_index/output_dim)

        eta = "ETA: " + str(round(sections_left * time_per_section, 1)) + " mins"
        draw_progress_bar((section_index + dim_offset) / total_sections, eta)

        section_index += output_dim

    # Fill in slices in the end
    end_aligned = len(file_names) - input_dim + dim_offset

    # Read section of folder
    section = read_folder_section(input_folder, end_aligned, end_aligned + input_dim).astype('float32')

    # Make the volume pixel intensity between 0 and 1
    section_vol = section / (2 ** 16 - 1)

    section_seg = helper_segment_section(model, section_vol)

    write_folder_section(output_folder, file_names, end_aligned, section_seg)

    total_time = "Total: " + str(round((time.time()/60) - start_time, 1)) + " mins"
    draw_progress_bar(1, total_time)
    print("\n")

def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)

"""
Helper function for segment_brain. Takes in a section of the brain 

*** Note: Due to the network output shape being smaller than the input shape, the edges will not be used  

@param models: network models
@param section: a section of the entire brain
"""


def helper_segment_section(model, section):
    # List of bottom left corner coordinate of all input chunks in the section
    coords = []

    # Pad the section to account for the output dimension being smaller the input dimension
    # temp_section = np.pad(section, ((0, 0), (dim_offset, dim_offset),
    #                                 (dim_offset, dim_offset)), 'constant', constant_values=(0, 0))
    temp_section = np.pad(section, ((0, 0), (dim_offset, dim_offset),
                                    (dim_offset, dim_offset)), 'edge')

    # Add most chunks aligned with top left corner
    for x in range(0, temp_section.shape[1] - input_dim, output_dim):
        for y in range(0, temp_section.shape[2] - input_dim, output_dim):
            coords.append((0, x, y))

    # Add bottom side aligned with bottom side
    for x in range(0, temp_section.shape[1] - input_dim, output_dim):
        coords.append((0, x, temp_section.shape[2]-input_dim))

    # Add right side aligned with right side
    for y in range(0, temp_section.shape[2] - input_dim, output_dim):
        coords.append((0, temp_section.shape[1]-input_dim, y))

    # Add bottom right corner
    coords.append((0, temp_section.shape[1]-input_dim, temp_section.shape[2]-input_dim))

    coords = np.array(coords)
    # List of cropped volumes that the network will process
    batch_crops = np.zeros((batch_size, input_dim, input_dim, input_dim))
    # List of coordinates associated with each cropped volume
    batch_coords = np.zeros((batch_size, 3), dtype="int")

    # Keeps track of which coord we are at
    i = 0

    # Generate dummy segmentation
    seg = np.zeros(temp_section.shape).astype('float32')

    # Loop through each possible coordinate
    while i < len(coords):

        # Fill up the batch by skipping chunks below the threshold
        batch_count = 0
        while i < len(coords) and batch_count < batch_size:
            (z, x, y) = coords[i]

            # Get the chunk associated with the coordinate
            test_crop = temp_section[z:z + input_dim, x:x + input_dim, y:y + input_dim]

            # Only add chunk to batch if its max value is above threshold
            # (Avoid wasting time processing background chunks)
            if np.max(test_crop) > threshold:
                batch_coords[batch_count] = (z, x, y)
                batch_crops[batch_count] = test_crop
                batch_count += 1
            i += 1

        # Once the batch is filled up run the network on the chunks
        batch_input = np.reshape(batch_crops, batch_crops.shape + (1,))
        output = np.squeeze(model.predict(batch_input)[:, :, :, :, [0]])

        # Place the predictions in the segmentation
        for j in range(len(batch_coords)):
            (z, x, y) = batch_coords[j] + dim_offset
            seg[z:z + output_dim, x:x + output_dim, y:y + output_dim] = output[j]

    cropped_seg = seg[:, dim_offset: dim_offset + section.shape[1], dim_offset: dim_offset + section.shape[2]]
    return cropped_seg
