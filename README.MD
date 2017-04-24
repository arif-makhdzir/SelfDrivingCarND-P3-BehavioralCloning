<h2>Behavioral Cloning</h2> 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
<h4>Files Submitted & Code Quality</h4>

<h4>1. Submission includes all required files and can be used to run the simulator in autonomous mode</h4>

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

<h4>2. Submission includes functional code</h4>
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

<h4>3. Submission code is usable and readable</h4>

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

<h3>Model Architecture and Training Strategy</h3>

<h4>1. An appropriate model architecture has been employed</h4>
My model is based on Comma.AI's end-to-end self driving car convolution neural network: https://github.com/commaai/research/blob/master/train_steering_model.py 

I chose Comma.AI's model as it is a proven model that functions perfectly on a similar image input - in terms of size and features in the image.

Pre-processing
1. Cropping - Cut 70px top and 25px bottom. Rational: Unneeded area for learning, sky & hood, trees, make image smaller, faster training. and allow model to focus on what's important (the road & lines)
2. Resize - Resize to 160x40. Make smaller faster to train, and fits into the comma.ai model
3. Normalized - (x/255.0) - 0.5 to get mean close to 0, make model learns better and faster.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x40x3 RGB image   							| 
| Convolution 8x8     	| 4x4 stride, same padding, depth of 16 	|
| ELU					|												|
| Convolution 5x5     	| 2x2 stride, same padding, depth of 32 	|
| ELU					|												|
| Convolution 5x5     	| 2x2 stride, same padding, depth of 64 	|
| Flatten	      	|  				|
| Dropout		| Keep probability = 0.2       									|
| ELU					|				|
| Fully connected		| Output = 512        									|
| Dropout		| Keep probability = 0.5       									|
| ELU					|		
| Dropout	| Keep probability = 0.5        									|
| Fully connected		| Output = 1       									|

Augmentation
1. Using left & right images - steering coefficient of 0.229(+/-). This helps the model to learn how to steer to middle when its off center, this fakes a situation of return to center from the side.
2. Flip image horizontally on odd frame count - for every odd frame I flip image horizontally and change the sign of the steering angle. This helps the model to generalize to steering magnitude, not always turn left. generalize turning left & right. Since the track is counter clokwise with lots of left turns, required to learn right turn. Even though drive clockwise, still help to generalize.

Driving strategy
1. Off center move to center
2. Drive reverse track, more right turn. problem first time turn right kelaut.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten	      	| Output = 400 				|
| Fully connected		| Output = 120        									|
| RELU					|												|
| Dropout	| Keep probability = 0.5        									|
| Fully connected		| Output = 84        									|
| RELU					|												|
| Dropout	| Keep probability = 0.5        									|
| Fully connected		| Output = 43       									|
| Softmax				|  

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

