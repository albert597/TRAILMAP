from inference import *
from models import *
import sys
import os
import shutil

if __name__ == "__main__":

    base_path = os.path.abspath(__file__ + "/..")

    input_batch = sys.argv[1:]
    # Verify each path is a directory
    for input_folder in input_batch:
        if not os.path.isdir(input_folder):
            raise Exception(input_folder + " is not a directory. Inputs must be a folder of files. Please refer to readme for more info")

    # Load the network
    weights_path = base_path + "/data/model-weights/trailmap_model.hdf5"

    model = get_net()
    model.load_weights(weights_path)

    for input_folder in input_batch:

        # Remove trailing slashes
        input_folder = os.path.normpath(input_folder)

        # Output folder name
        output_name = "seg-" + os.path.basename(input_folder)
        output_dir = os.path.dirname(input_folder)

        output_folder = os.path.join(output_dir, output_name)


        # Create output directory. Overwrite if the directory exists
        if os.path.exists(output_folder):
            print(output_folder + " already exists. Will be overwritten")
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

        # Segment the brain
        segment_brain(input_folder, output_folder, model)

