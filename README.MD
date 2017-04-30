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

For the final model architecture, see next section.

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

<h4>2. Attempts to reduce overfitting in the model</h4>

i) The model contains dropout layers after the two fully connected layers in order to reduce overfitting (model.py lines 21). 

ii) The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16 for training & validation set split). 

iii) The track is counter clockwise, so there are way more left turns than right turns. In order to combat the model from overfitting to left turn steering action, flip horizontal data augmentation technique was used and driving the track reverse to capture more right turn data was done.

<h4>3. Model parameter tuning</h4>

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

<h4>4. Appropriate training data</h4>

Training data was chosen to keep the vehicle driving on the road. Firstly I combine Udacity's dataset with my own training data in order to come up with the final data set.

For my own training data, I used mainly a combination of center recovering from the left and right sides of the road and reverse track driving. These training data compliments Udacity's dataset well. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

<h5>Chosing model</h5>
My first step was to use a convolution neural network model similar to the Comma.AI's end-to-end self driving car convnet: https://github.com/commaai/research/blob/master/train_steering_model.py 

I chose Comma.AI's model as it is a proven model that functions perfectly on a similar image input - in terms of size and features in the image.

Note that Comma.AI's model already has dropout layers, so I just copy them and use the same keep probability hyperparameter

<h5>Preprocessing</h5>
First thing to do (as outlined in the lecture) is to normalize the images. This is done by adding a lambda layer that does the following operation: (x/255.0) - 0.5 (line 128)

<h5>First training attempt and evaluation</h5>
In order to gauge how well the model was working, I split my data set into training and validation set, with 80/20 split. I trained my model with Udacity's dataset, using Adam Optimizer (lr=0.0001) with batch size of 32 for 5 epochs. The end result was 0.2 loss on trainng set and 0.27 loss on validation set. I ran my model on the test track and it failed on the second turn. The model is underfitting, I need the training set loss to be lower (less than 0.05).

There are two ways to combat underfitting, one way is to have more useful data or to reduce the size of the model. The latter is not an option, as the model is of the right size for this problem, so I decided to focus on improving the data set. Also I decided to augment/improve udacity's dataset before adding my own dataset, as udacity's dataset is clean and probably is the best driving behavior I can find.

From inspection of Udacity's dataset, I find that quite a bit of the data are just going straight (steering angle of 0). The model needs to learn turning better, thus we need more data that contains non-zero steering angle.

I used a couple of strategy to add more non-zero steering angle data:

<h5>1. Adding left & right images data into the sample set</h5>
I augment the data by adding the left & right image data into the sample set. Now the question is what steering angle should I use for these images, to simulate a drive to center dataset? So I did a trial-and-error experiment below to find the best steering co-efficient:

| Steering angle co-efficient         		|     Observation of test track run	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.229         		| Steering turning a bit too sharp, too gittery even when driving straight  							| 
| 0.220     	| Model drives smoother on straight line, but fail to turn when corner too sharp 	|
| 0.227					|	Best result, quite smooth on straight line yet still able to turn at sharp corner											|

Result: Model is able to drive all left turn without problem, however when it encounters right turn it drives off the track. This means my model is not generalizing to turning right well-enough, so I need to augment/add more turning right data.

<h5>2. Adding traning data of center recovering from the left and right sides of the road</h5>
These data helps..

<h5>Flip horizontal & drive reverse track</h5>
1. Flip horizontal
In order to augment the data to get more right turns, I flip the image horizontal for every odd frame and reverse the steering co-efficient sign (line xx).

2. Drive reverse track
I collect more data of me drivign reverse track so that I get more right turn data point.

Model drives perfectly around the track for many laps.

----------------------------------------------------------

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

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
| Fully connected		| Output = 1       									| Output = 43       									|
| Softmax				|  


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

