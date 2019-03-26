# Facial-Keypoint-Detection
CNN for Facial Keypoints Detection Project [Udacity Computer Vision Nanodegree]

## Project Overview
### Project Description
Implement computer vision techniques and deep learning architectures to build a facial keypoint detection system. This sysytem can take in any image with faces, and predicts the location of 68 distinguishing keypoints around the eyes, nose, and mouth on a face.

### Project Procedure
- Import datasets
  - Dog dataset
  - Human dataset
- Preprocess the data with shape (batch, rows, columns, channels)
- Write detectors
   - face detector
   - dog detector
- Create a CNN
- Compile the model
  - train, test
- Write a algorithm for the dog breeds classifier
- Turn the code into a web app using Flask

### Project Results
The model can successfully detect dogs and faces and make predictions on the given image.
If detect a dog, the model will identify an estimate of the dog's breed. If detect any face, the code will identify the resembling dog breed for each face and overlay filters with dog's ears, nose and tongue on it.

## Getting Started
### Installing
#### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
#### Notebooks

1. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd
cd Facial_Keypoint_Detection
```

2. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

3. Once you open any of the project notebooks, make sure you are in the correct `cv-nd` environment by clicking `Kernel > Change Kernel > cv-nd`.

## Data
All of the data you'll need to train a neural network is in the the subdirectory `data`.
