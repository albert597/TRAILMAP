
# TRAILMAP - Extended Version

![Figure Image](https://www.biorxiv.org/content/biorxiv/early/2019/10/21/812644/F3.large.jpg?width=800&height=600&carousel=1)

This is a software package to extract axonal data from cleared brains as demonstrated in [Mapping Mesoscale Axonal Projections in the Mouse Brain Using A 3D Convolutional Network](https://www.biorxiv.org/content/10.1101/812644v1.full) Friedmann D, Pun A, et al. While it was trained and tested on axons in iDISCO-cleared samples imaged by lightsheet microscopy, it works well at identifying many types of filamentous structures in 3D volumes. Instructions are included for segmenting your data with our best existing model, but we also provide guidance on transfer learning with your own annotated data.

These instructions are written to be simple enough to be useful for novice users—not just of machine learning tools, but even if this is your first time using python or linux, you can hopefully follow along. If you prefer a less verbose set of instructions, there is also a shorter readme available.

[Readme Short Version](../master/README.md)

## Getting Started - Installation

These steps assume that you are starting with a fresh install of Ubuntu and are a novice to both python and linux. As such, many of these steps can be skipped if you already have things installed.

### Prerequisites

##### Hardware requirements:
* While the network portion of TrailMap operates on your 3D volume in chunks and has minimal hardware requirements, visualizations benefit from having sufficient RAM to hold the whole sample volume at once. Depending on your brain file size, TrailMap will take 8-16GB of RAM. Opening a 16-bit volume covering a half-brain (as seen in the publication) requires ~24 GB. In practice, 64GB is sufficient, but 128GB+ provides greater flexibility when opening multiple volumes at once.

* A Nvidia GPU with CUDA support. The network presented here was trained with a 1080Ti with 11GB GDDR5X Memory. 

* You need to also install Nvidia Driver 390, CUDA 9.0, and CUDNN 7.0 for tensorflow to use your GPU. [Guide on installation for CUDA and CUDNN](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)


##### Software requirements:
* Python 3.6

Many different opinions exist on the best way to install and manage python environments. If you’re new to this, the following steps will get the job done while being flexible for future changes.

Current versions of Ubuntu ship with python included. Check this in a Terminal window with:

```
sudo apt update
```
confirm “yes” as needed by typing “y” and hitting "return"

Check that everything is ok with:

```
python3 -V
```
which will report back to you the version you have installed.

Install ‘pip’ – a tool for installing python packages
```
sudo apt install -y python3-pip
```

Install 'virtual environments' - a tool for creating isolated dependency environments
```
sudo apt install python3-venv
```

##### Making a ‘virtual environment’ for TrailMap so that you can manage its packages separate from the system’s packages:

Create an empty folder somewhere safe named ‘environments’ (or any name you like)

In ‘Terminal’, enter into this folder, eg. 
```
cd /home/USERNAME/Documents/environments
```
then, 
```
python3 -m venv trailmap-env
```
Check that the folder has been populated by new folders called “bin,” “include,” “lib,” etc.

Any time you would like to use this environment, run

```
source home/USERNAME/Documents/environments/trailmap-env/bin/activate
```
and you will see (trailmap-env) appear on the left edge of the line to indicate that you are in the correct environment.


##### Dependencies
From within this environment, use ‘pip’ to install the following packages that are required for TrailMap to run:

```
pip3 install tensorflow-gpu==1.9
pip3 install opencv-python==4.1.2.30
pip3 install pillow==7.0
pip3 install numpy==1.16
pip3 install h5py==2.1
```

##### TrailMap:

You must have git installed along with git-lfs

To install git and enable git-lfs, run the following commands
```
sudo apt install git
git lfs install
```

To download the required files, run
```
git clone https://github.com/AlbertPun/TRAILMAP.git
```

From Terminal, enter into the TRAILMAP directory

```
cd /home/USERNAME/Documents/TRAILMAP
```

and you should see:

```
(trailmap-env) USERNAME@computername:~/home/USERNAME/Documents/TRAILMAP$
```

This is how it should look each time you would like to use TrailMap. Once installed, each time you start from a fresh Terminal window, you just need to run the following two commands to get back to this point:

```
source home/USERNAME/Documents/environments/tensorflow-env/bin/activate
cd /home/USERNAME/Documents/TRAILMAP
```


## Inference - Using our model to segment your data

##### File structure of your image data:

Your brain volume must be in the form of single-channel 2D image slices in a folder where the slice order is determined by the alphanumerical order of the image file names. See TRAILMAP/data/testing/example-chunk for an example. If you have a TIFFstack that is a single file, you can use FIJI to generate the correct file structure. (File>Save As>Image Sequence)

To run the network, enter your virtual environment and the TRAILMAP folder as described above and run:
```
python3 segment_brain_batch.py input_folder1 input_folder2 input_folder3 
```

where “input_folderX” is the absolute path to the folder of the brain you would like to process. A fast and easy way to populate these fields is to right-click the folder containing your images, select copy, and in the Terminal window right-click and select ‘Paste Filename.’ 

If you receive an error about running out of memory, depending on the amount of GPU RAM you have available, you may need to lower the “batch_size” value in the segment_brain.py file.


The program ‘segment_brain_batch.py’ will create a new folder named “seg-input_folderX " in the same directory as the input_folder. 

##### Example: 

```
python3 segment_brain_batch.py data/testing/example-chunk
```

You should see a message saying "Adding visible gpu devices: 0" which would indicate tensorflow is using your gpu. If you do not see this message, please check your CUDA and CUDNN installations for your GPU. The segmentation program should take at most 1-2 minutes for the example-chunk.

This output will be the same size as your input data, and corresponds to the probability that each voxel represents an axon (values from 0 to 1, in 32 bit format). Opening the folder as a ‘Virtual Stack’ in FIJI may be required if your volume is large and your computer does not have sufficient RAM.


## Training – add your own labels to improve performance

If you would like to do transfer learning with your own examples, you must label some of your own data. The following steps will walk you through our strategy to create these training examples efficiently. 

An example of the end goal can be seen in the pair of image files found at: 
* data/training/training-original/labels/training_example_label.tif
* data/training/training-original/volumes/training_example_volume.tif

The ‘volume’ is a small chunk of raw data; we typically crop to a volume of ~128-200 pixels on each edge. The ‘label’ is an 8-bit TIFF of the same dimensions, primarily consisting of “0’s”, but with individual slices annotated at a spacing of ~20-40 slices in the Z dimension.

When labeling, the end product will have the following legend. Each annotated slice must be fully labeled (all voxels) with:
* **1** - background
* **2** - axon
* **3** - artifacts (objects that look like axons and should be avoided)
* **4** - the edge of the axon (should be programmatically inserted)

All other slices should be labeled with 
* **0** - unlabeled 

##### Our recommended annotation strategy:

Create cropped 3D subvolumes of raw data ~128-200 px on an edge and save them in the folder in data/training/training-original/volumes. Save these as single TIFF files. With one of these images open in FIJI, start up the Segmentation Editor plugin (Plugins>Segmentation>SegmentationEditor). Here, you can scroll in Z to a given slice and begin drawing on the image (right-click the “oval selection” button to switch to “selection brush tool” then double click to set brush size). Selected pixels (outlined in yellow) can be iteratively added and subtracted to the “Labels” window by clicking on the “+” and “-“ buttons. A second label (for artifacts) can be generated by right-clicking on the list at left. Label the axons and artifacts on each of a few slices (but you must label all axons/artifacts on that slice). **The “background” and “edge” labels will be added automatically later.**

When finished, select ‘Ok’ and save the .labels image as a TIFF in data/training/training-original/temp. Keep your file names consistent: TrailMap will determine each volume's matching annotation file by alphanumerically sorting the files in the ‘volumes’ and ‘labels’ folders and assuming the order is the same in each.

Add ‘background’ (1) and ‘edges’ (4) labels and increment ‘axon’ (1 to 2) and ‘artifact’ (2 to 3) labels by running:

```
python3 prepare_data.py "process_labels" PATH_TO_LABEL_FOLDER
```

where PATH_TO_LABEL_FOLDER is the folder containing all your labeled tiff volumes (eg. data/training/training-original/temp). This will create a folder ‘edge-ORIGINAL_FOLDER_NAME’ with the new edge label added to each labeled volume and the correct values for axons and artifacts. Check in FIJI that each labeled slice has the background, axon, artifact, and edge values as 1, 2, 3, and 4 respectively. If so, move these files to data/training/training-original/labels.

##### Training Set:

After you have placed your labeled examples in the training-original folder you can populate the training-set folder by running

```
python3 prepare_data.py "generate_training_set" NUM_EXAMPLES
```

NUM_EXAMPLES is an optional parameter that determines the numbers of crops to make in total using a round robin strategy from the ‘training-original’ folder. If not specified, the default value is set to 100 * number of images in the folder.

##### Validation Set:

You must also include a validation set to judge your network's performance on the new data. You must put your validation examples in ‘data/training/validation-original’ with the same labeling technique as the training. You can then populate the validation-set folder by running

```
python3 prepare_data.py "generate_validation_set" num_examples
```

##### Training:

After populating the training-set folder, you may start the transfer learning. This will require you to tune the parameters in train.py and volume_data_generator.py

There are some default parameters for training, but you will likely need to tune this depending on how different your own training set is to our data and if you need to do any augmentation. The VolumeDataGenerator class is provided to handle basic operations (the train.py contains this class and more specific information in the comments). This follows the same paradigm as Tensorflow's ImageDataGenerator.

Try different augmentations (eg. Scale, Normalization, etc.) first before changing parameters in model.py (eg. learning rate, loss weights).

After you have populated the training-set folder and tuned parameters, start training with:

```
python3 train.py
```

Note: Depending on the amount of GPU memory, you may need to reduce the batch_size in the train.py file. You may also change steps_per_epoch = floor(NUM_TRAINING_EXAMPLES/batch_size) and validation_steps = floor(NUM_VALIDATION_EXAMPLES/batch_size)

This will load in our current model and start training the model on your own data. Checkpoints are saved to data/model-weights at the end of each epoch along with tensorboard logs in data/tf-logs

##### Tensorboard – watching statistics evolve during training

In a new Terminal window, activate the same python environment and run:

```
tensorboard --logdir=/PATH_TO_TF-LOG-FILE
```

Copy the link it outputs (eg. http://COMPUTER:####) and paste it into your web browser.

If you are new to machine learning, some values to observe are ‘epoch_axon_recall’ and ‘epoch_val_loss’ which give you an idea of how well the network is finding axons at this moment in the training (recall) and whether or not it is overfitting to your data (if the val_loss is increasing while the recall is also increasing).

You can try the following example, 

```
tensorboard --logdir='data/tf-logs/example-logs'
```

##### Testing the performance of your new model

Select a model from data/model-weights to test and copy its name into line #19 of segment_brain_batch.py before proceeding with the instructions seen above for “Inference.” Training with new parameters will overwrite existing models with unchanged file names, so if you identify a particularly good model, be sure to change its name before running train.py again.

## Authors

* **Albert Pun**
* **Drew Friedmann**

## License

This project is licensed under the MIT License

## Acknowledgments

* Research sponsored by Liqun Luo's Lab


