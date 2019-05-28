## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I try out my model on images of German traffic signs that I find on the web.

The steps of this project are listed below. You can have a look at [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb) for the code.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Dependencies
---
This project requires Python 3.5 and the following Python libraries installed:

* Jupyter
* NumPy
* SciPy
* scikit-learn
* TensorFlow
* Matplotlib
* Pandas (Optional)

Run this command at the terminal prompt to install OpenCV. Useful for image processing:

```
conda install -c https://conda.anaconda.org/menpo opencv3
```

Data Set Summary & Exploration
---
__1. Basic summary of the data set.
In order to calculate summary statistics for the data set, I used numpy library. The results are as follows:__

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32x32x3. This means that each picture is 32 pixels wide, 32 pixels tall, and has 3 color channels.
* The number of unique classes/labels in the data set is 43.

__2. Exploratory visualization of the dataset.__

I plot some traffic signs randomly. As you can see from the images below, the lighting conditions under which the phtos were taken vary drastically. Some of the photos are hard to tell apart because the lighting was too dark. This can be a big challenge to train the network.

![data_set](./visualizations/matrix.png)

Also the number of training examples for each class varies. The histogram showing the number of training examples for each class. The imbalance data set is another thing need to be fixed. In order to solve this issue, I balanced the classes in such a way that all of them have equal number of images.

![data_set](./visualizations/hist-unprocessed.png)

Design and Test a Model Architecture
---
__1. Image preprocessing.__
* Normalization:
Normalizing all pictures  by substracting mean value and dividing by the standard deviation of the mean value. This normalization was performed for each of 3 of the color channels. (This approach was inspired by the [VGG16](https://arxiv.org/pdf/1409.1556.pdf) paper)
An important consideration was data leakage, so the mean and SD values used for all normalization were calculated only based on the training data set, and not validation or test data.

* Augmentation:
To make class balance, I used CV2 package to rotation, translation and shear to generate new images. Resulting transformed images can be seen below.

![data_set](./visualizations/matrix-aug.png)

After augmentation/balancing, each class has 2010 images, which is how many images the largest class had before augmentation.

![data_set](./visualizations/hist-processed.png)
