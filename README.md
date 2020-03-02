
# TRAILMAP

![Figure Image](https://www.biorxiv.org/content/biorxiv/early/2019/10/21/812644/F3.large.jpg?width=800&height=600&carousel=1)

This is a software package to extract axonal data from cleared brains as demonstrated in [Mapping Mesoscale Axonal Projections in the Mouse Brain Using A 3D Convolutional Network](https://www.biorxiv.org/content/10.1101/812644v1.full) Friedmann D, Pun A, et al. While it was trained and tested on axons in iDISCO-cleared samples imaged by lightsheet microscopy, it works well at identifying many types of filamentous structures in 3D volumes. Instructions are included for segmenting your data with our best existing model, but we also provide guidance on transfer learning with your own annotated data.

There are two readmes—this one includes basic instructions and descriptions for getting TrailMap up and running. The second is written to be simple enough to be useful for novice users—not just of machine learning tools, but even if this is your first time using python or linux, you can hopefully follow along. 

[Readme Extended Version](../master/README-EXTENDED.md)

## Getting Started - Installation

You must have git installed along with git-lfs

To download the required files, run
```
git clone https://github.com/AlbertPun/TRAILMAP.git
```

From Terminal, enter into the TRAILMAP directory

```
cd /home/USERNAME/Documents/TRAILMAP
```

Due to the large size of the model, you must enable git-lfs in this directory by running
```
git lfs install
```

If git-lfs does not work for you (e.g. error messages with the model being corrupted), we have also provided a google drive link to the model weights [here](https://drive.google.com/file/d/1-G-hhH0F0SjzVDDtEsWtVFA-UCpCVE3m/view?usp=sharing). Drag the downloaded file into the TRAILMAP/data/model-weights folder.

### Prerequisites

##### Hardware requirements:
* While the network portion of TrailMap operates on your 3D volume in chunks and has minimal hardware requirements, visualizations benefit from having sufficient RAM to hold the whole sample volume at once. Depending on your brain file size, TrailMap will take 8-16GB of RAM. Opening a 16-bit volume covering a half-brain (as seen in the publication) requires ~24 GB. In practice, 64GB is sufficient, but 128GB+ provides greater flexibility when opening multiple volumes at once.
* A Nvidia GPU with CUDA support. The network presented here was trained with a 1080Ti with 11GB GDDR5X Memory. 
* You need to also install the  Nvidia Driver Verision > 418 with CUDA 10.1 and CUDNN 7.6 to use your GPU. [Guide on installation for CUDA and CUDNN](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)

##### Software requirements:
* Python 3.7

We **highly** recommend you use Anaconda to manage your packages because it is by far the easiest way to install cuda and cudnn with the correct versions. Refer to the [Readme Extended Version](../master/README-EXTENDED.md) for step by step instructions on how to do this with Anaconda

```
tensorflow-gpu==2.1
opencv==3.4
pillow==7.0
numpy==1.18
h5py==2.1
```

## Inference

Data structure: Your brain volume must be in the form of 2D image slices in a folder where the slice order is determined by the alphanumerical order of the image file names. See TRAILMAP/data/testing/example-chunk for an example

To run the network cd into the TRAILMAP directory and run 
```
python3 segment_brain_batch.py input_folder1 input_folder2 input_folder3 
```

where “input_folderX” is the absolute path to the folder of the brain you would like to process. Note: Depending on the amount of GPU memory, you may need to lower the batch_size in the segment_brain.py file.

The program will create a folder named “seg-input_folderX " in the same directory as the input_folder

To segment an example chunk run
```
python3 segment_brain_batch.py data/testing/example-chunk
```
Note: Depending on the amount of GPU memory, you may need to lower the batch_size in the segment_brain.py file.

## Training

If you would like to do transfer learning with your own examples, you must label some of your own data. The network takes in cubes with length of 64 pixels, so the training examples must be of this size. 

When labeling the cube, you must follow the following legend. You only need to label individual slices, spaced every 20-40 slices, with the labels:
* **1** - background
* **2** - axon
* **3** - artifacts (objects that look like axons and should be avoided)
* **4** - the edge of the axon (should be programmatically inserted)

All other slices should be labeled with 
* **0** - unlabeled 

Our strategy, with provided utilities programs: 

The basic procedure is to hand label cubes of length ~128-200 px (placed in the folders in data/training/training-original/labels and data/training/training-original/volumes) and crop out many cubes of length 64px (into the folder /training-set/). While we recommend this strategy, you may use your own strategy of populating the training-set folder if you wish. We recommend using [ImageJ's Segmentation Editor Plugin](https://imagej.net/Segmentation_Editor) to label your data.


TrailMap will determine a volume's matching label by sorting files in the ‘volumes’ and ‘labels’ folders alphabetically and assuming the label and volume at the same index are pairs.

We have included a script to add the edge label next to an axon label. After you have labeled axons in your volumes, run
```
python3 prepare_data.py "process_labels" PATH_TO_LABEL_FOLDER
```

where PATH_TO_LABEL_FOLDER is the folder containing all your labeled tiff volumes. This will create a folder edge-ORIGINAL_FOLDER_NAME with the new edge label added to each labeled volume.

Fully labeled examples are show here:
* data/training/training-original/labels/training_example_label.tif
* data/training/training-original/volumes/training_example_volume.tif

After you have placed your labeled examples in training-original folder you can populate the training-set folder by running
```
python3 prepare_data.py "generate_training_set" NUM_EXAMPLES
```

NUM_EXAMPLES is an option parameter that determines the numbers of crops to make in total using a round robin strategy from the training-original folder. If not specified, the default value is set to 100 * NUM_TRAINING_ORIGINAL_EXAMPLES

You must also include a validation set to judge your network's performance on the new data. You must put your validation examples in the data/training/validation-original with the same labeling technique as the training. You can then populate the validation-set folder by running
```
python3 prepare_data.py "generate_validation_set" num_examples
```

After generate_training_set has populated the training-set folder, you may start the transfer learning. This will require you to tune the parameters in train.py

There are some default parameters for training, but you will likely need to tune this depending on how different your own training set is to our data and if you need to do any augmentation.

A VolumeDataGenerator class is provided that handles basic operations (the train.py contains this class and more specific information in the comments). This follows the same paradigm as Tensorflow's ImageDataGenerator.

After you have populated the training-set folder and tuned parameters, start training with:
```
python3 train.py
```

Note: Depending on the amount of GPU memory, you may need to reduce the batch_size in the train.py file. You may also change steps_per_epoch = floor(NUM_TRAINING_EXAMPLES/batch_size) and validation_steps = floor(NUM_VALIDATION_EXAMPLES/batch_size)
This will load in the current model and start training the model on your own data. Checkpoints are saved to data/model-weights at the end of each epoch along with tensorboard logs in data/tf-logs


## Authors

* **Albert Pun**
* **Drew Friedmann**

## License

This project is licensed under the MIT License

## Acknowledgments

* Research sponsored by Liqun Luo's Lab

