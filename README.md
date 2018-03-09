# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Center_Lane_Driving.jpg "Center Lane Driving"
[image2]: ./examples/Left_to_Center_1.jpg "Recovery Image 1"
[image3]: ./examples/Left_to_Center_2.jpg "Recovery Image 2"
[image4]: ./examples/Left_to_Center_3.jpg "Recovery Image 3"
[image5]: ./examples/orginal.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 recording of vehicle driving autonomously for one lap around the track
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 53-74) 

The model includes RELU layers to introduce nonlinearity (code line 57-64), and the data is normalized in the model using a Keras lambda layer (code line 54) and cropped using a Keras cropping layer (code line 55). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after every RELU activation in order to reduce overfitting (model.py lines 57-65). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 13-47). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I created the training data by driving 2 laps of center lane driving and 1 lap of left/right sides recovering. Plus, I also made additional driving data from the bridge since the materials is differnt from other road.

I used all 3 cameras from left, center and right to balance the training data more and make it more general.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a nural networks that finds the minimum squre error from the training image data. So that the model can determine which steering angle to use to stay in the center of the lane and also be able to recover from the sides if it drifts.

My first step was to use a convolution neural network model similar to the one from Nvidia paper that showed the course video. I thought this model might be appropriate because this model was designed for processing images captured from self-driving cars. And the model turned out to be working pretty well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to have dropout layer after every RELU activation. 

Then I tuned the keep probability from range 0.5-1.0 and found 0.5 to be working just fine for my training procedure. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as the nridge and the sharp turn after the bridge. To improve the driving behavior in these cases, I generated more data to those specific locations.

One more thing worth mentioning is that I used cv2.imread() in my model which reads in BGR color space but the drive.py takes RGB image and this causes some confusion for the model. I thus modified the model and change the color space to RGB to make it consistent thoughout the pipeline.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 53-70) consisted of a convolution neural network with the following layers and layer sizes:

 1. Lambda layer: normalize the input image
 2. Cropping layer: crop out top and bottom of the image to exclude unwanted information such as sky, hills and the car hood.
 3. Convolution layer 1: filter size = 5x5, depth = 24, stride = 2, RELU activation
 4. Dropout layer 1: keep_prob = 0.5
 5. Convolution layer 2: filter size = 5x5, depth = 36, stride = 2, RELU activation
 6. Dropout layer 2: keep_prob = 0.5
 7. Convolution layer 3: filter size = 5x5, depth = 48, stride = 2, RELU activation
 8. Dropout layer 3: keep_prob = 0.5
 9. Convolution layer 4: filter size = 3x3, depth = 64, stride = 1, RELU activation
10. Dropout layer 4: keep_prob = 0.5
11. Convolution layer 5: filter size = 3x3, depth = 64, stride = 1, RELU activation
12. Dropout layer 5: keep_prob = 0.5
13. Flatten layer: flatten to 1 col
14. Dense layer 1: dense to depth = 100
15. Dense layer 2: dense to depth = 50
16. Dense layer 3: dense to depth = 10
17. Dense layer 4 (Output layer): dense to depth = 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from drifting off the track. These images show what a recovery looks like starting from left side back to center :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Besides that center images, I also include left and right images from the side cameras and flip them to augment more training data. Those images can also be seen as recovering from the side, as I added +/- 0.2 degree to the steering angle to compensate.

Up to this point, I split the data set to 80%/20% for training and validation datasets.

To augment the data set, I also flipped images and angles thinking that this would generalize the model to not bias to one direction (track one has a left turn bias to it) For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 38,796 number of data points. I then preprocessed this data by changing the color space from BGR to RGB, then flip the image and steering angle. For images from left and right cameras, I also added 0.2 degree angle compensation.

The final data sets fed into the model totals 77,592 number of data points (training = 62,074, validation = 15,518).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by that more ephchs do not reduce the mse much but take a lot more time to train. I used an adam optimizer so that manually training the learning rate wasn't necessary.
