# Define a Network Architecture
# The various layers that make up any neural network are documented, here. For a convolutional neural network, we'll use a simple series of layers:
# Convolutional layers
# Maxpooling layers
# Fully-connected (linear) layers

# To define a neural network in PyTorch, you'll 
# 1. create and name a new neural network class
# 2. define the layers of the network in a function __init__ 
# 3. define the feedforward behavior of the network that employs those initialized layers in the function forward, which takes in an input image tensor, x. 
# The structure of such a class, called Net is shown below.
# Note: During training, PyTorch will be able to perform backpropagation by keeping track of the network's feedforward behavior 
# and using autograd to calculate the update to the weights in the network.

import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


class Net(nn.Module):

    def __init__(self, n_classes):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 32 output channels/feature maps
        # Load the training image
        image = cv2.imread('./images/face.jpeg')

        # Convert the training image to RGB
        training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the training image to gray Scale
        training_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Display the images
        plt.subplot(121)
        plt.title('Original Training Image')
        plt.imshow(training_image)
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # fully-connected layer
        # 32*4 input size to account for the downsampled image size after pooling
        # num_classes outputs (for n_classes of image data)
        self.fc1 = nn.Linear(32*4, n_classes)

    # define the feedforward behavior
    def forward(self, x):
        # one conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))

        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        # linear layer 
        x = F.relu(self.fc1(x))

        # final output
        return x

# instantiate and print your Net
n_classes = 20 # example number of classes
net = Net(n_classes)
print(net)

# Let's go over the details of what is happening in this code.
# Define the Layers in __init__
# Convolutional and maxpooling layers are defined in __init__:
# 1 input image channel (for grayscale images), 32 output channels/feature maps, 3x3 square convolution kernel
# self.conv1 = nn.Conv2d(1, 32, 3)

# maxpool that uses a square window of kernel_size=2, stride=2
# self.pool = nn.MaxPool2d(2, 2)      
# Refer to Layers in forward
# Then these layers are referred to in the forward function like this, in which the conv1 layer has a ReLu activation applied to it before maxpooling is applied:
x = self.pool(F.relu(self.conv1(x)))
# Best practice is to place any layers whose weights will change during the training process in __init__ and refer to them in the forward function; any layers or functions that always behave in the same way, such as a pre-defined activation function, may appear in the __init__ or in the forward function; it is mostly a matter of style and readability.

# import the usual resources
import matplotlib.pyplot as plt
import numpy as np

# import utilities to keep workspaces alive during model training
from workspace_utils import active_session

# watch for any changes in model.py, if it changes, re-load it automatically
%load_ext autoreload
%autoreload 2

## TODO: Define the Net in models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from models import Net

net = Net()
print(net)

# Transform the dataset
# To prepare for training, create a transformed dataset of images and keypoints.
# TODO: Define a data transform
# In PyTorch, a convolutional neural network expects a torch image of a consistent size as input. For efficient training, and so your model's 
# loss does not blow up during training, it is also suggested that you normalize the input images and keypoints. 
# The necessary transforms have been defined in data_load.py and you do not need to modify these; take a look at this file 
# (you'll see the same transforms that were defined and applied in Notebook 1).
# To define the data transform below, use a composition of:
# Rescaling and/or cropping the data, such that you are left with a square image (the suggested size is 224x224px)
# Normalizing the images and keypoints; turning each RGB image into a grayscale image with a color range of [0, 1] and transforming the given keypoints into a range of [-1, 1]
# Turning these images and keypoints into Tensors
# These transformations have been defined in data_load.py, but it's up to you to call them and create a data_transform below. This transform will be applied to the training data and, later, the test data. It will change how you go about displaying these images and keypoints, but these steps are essential for efficient training.
# As a note, should you want to perform data augmentation (which is optional in this project), and randomly rotate or shift these images, a square image size will be useful; rotating a 224x224 image by 90 degrees will result in the same shape of output.

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = None

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',
                                             root_dir='/data/training/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())
    
# Batching and loading data
# Next, having defined the transformed dataset, we can use PyTorch's DataLoader class to load the training data in batches of whatever size as well as to shuffle the data for training the model. You can read more about the parameters of the DataLoader, in this documentation.
# Batch size
# Decide on a good batch size for training your model. Try both small and large batch sizes and note how the loss decreases as the model trains. Too large a batch size may cause your model to crash and/or run out of memory while training.
# Note for Windows users: Please change the num_workers to 0 or you may face some issues with your DataLoader failing.   

# load training data in batches
batch_size = 10

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)


# Before training
# Take a look at how this model performs before it trains. You should see that the keypoints it predicts start off in one spot and don't match 
# the keypoints on a face at all! It's interesting to visualize this behavior so that you can compare it to the model after training and see how 
# the model has improved.
# Load in the test dataset
# The test dataset is one that this model has not seen before, meaning it has not trained with these images. We'll load in this test data and before 
# and after training, see how your model performs on this set!
# To visualize this test data, we have to go through some un-transformation steps to turn our images into python images from tensors and to turn our 
# keypoints back into a recognizable range.                            

# load in the test data, using the dataset class
# AND apply the data_transform you defined above

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='/data/test_frames_keypoints.csv',
                                             root_dir='/data/test/',
                                             transform=data_transform)

# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)

# Apply the model on a test sample
# To test the model on a test sample of data, you have to follow these steps:
# Extract the image and ground truth keypoints from a sample
# Wrap the image in a Variable, so that the net can process it as input and track how it changes as the image moves through the network.
# Make sure the image is a FloatTensor, which the model expects.
# Forward pass the image through the net to get the predicted, output keypoints.
# This function test how the network performs on the first batch of test data. It returns the images, the transformed images, 
# the predicted keypoints (produced by the model), and the ground truth keypoints.

# test the model on a batch of test images

def net_sample_output():
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

# Debugging tips
# If you get a size or dimension error here, make sure that your network outputs the expected number of keypoints! Or if you get a Tensor type error, look into changing the above code that casts the data into float types: images = images.type(torch.FloatTensor).
# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints
test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())            

# Visualize the predicted keypoints
# Once we've had the model produce some predicted output keypoints, we can visualize these points in a way that's similar to how we've displayed this data before,
# only this time, we have to "un-transform" the image/keypoint data to display it.
# Note that I've defined a new function, show_all_keypoints that displays a grayscale image, its predicted keypoints and its ground truth keypoints 
# (if provided).

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
        	
# Un-transformation
# Next, you'll see a helper function. visualize_output that takes in a batch of images, predicted keypoints, and ground truth keypoints and displays a set of those images and their true/predicted keypoints.
# This function's main role is to take batches of image and keypoint data (the input and output of your CNN), and transform them into numpy images and un-normalized keypoints (x, y) for normal display. The un-transformation process turns keypoints and images into numpy arrays from Tensors and it undoes the keypoint normalization done in the Normalize() transform; it's assumed that you applied these transformations when you loaded your test data.

# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)
        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image
        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
# undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()
    
# call it
visualize_output(test_images, test_outputs, gt_pts)

# Training
# Loss function
# Training a network to predict keypoints is different than training a network to predict a class; instead of outputting a distribution of classes 
# and using cross entropy loss, you may want to choose a loss function that is suited for regression, which directly compares a predicted value and target value.
# Read about the various kinds of loss functions (like MSE or L1/SmoothL1 loss) in this documentation.
# TODO: Define the loss and optimization
# Next, you'll define how the model will train by deciding on the loss function and optimizer.

## TODO: Define the loss and optimization
import torch.optim as optim

criterion = None
optimizer = None

# Training and Initial Observation
# Now, you'll train on your batched training data from train_loader for a number of epochs. 
# To quickly observe how your model is training and decide on whether or not you should modify it's structure or hyperparameters, 
# you're encouraged to start off with just one or two epochs at first. As you train, note how your the model's loss behaves over time: 
# does it decrease quickly at first and then slow down? Does it take a while to decrease in the first place? What happens if you change the batch size 
# of your training data or modify your loss function? etc. 
# Use these initial observations to make changes to your model and decide on the best architecture before you train for many epochs and create a final model.

def train_net(n_epochs):

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')

# train your network
n_epochs = 1 # start small, and increase when you've decided on your model structure and hyperparams

# this is a Workspaces-specific context manager to keep the connection
# alive while training your model, not part of pytorch
with active_session():
    train_net(n_epochs)

# Test data
# See how your model performs on previously unseen, test data. We've already loaded and transformed this data, similar to the training data. Next, run your trained model on these images to see what kind of keypoints are produced. You should be able to see if your model is fitting each new face it sees, if the points are distributed randomly, or if the points have actually overfitted the training data and do not generalize.
# get a sample of test data again
test_images, test_outputs, gt_pts = net_sample_output()
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())                    

## TODO: visualize your test output
# you can use the same function as before, by un-commenting the line below:
# visualize_output(test_images, test_outputs, gt_pts)
# Once you've found a good model (or two), save your model so you can load it and use it later!
# Save your models but please delete any checkpoints and saved models before you submit your project otherwise your workspace may be too large to submit.

## TODO: change the name to something uniqe for each new model
model_dir = 'saved_models/'
model_name = 'keypoints_model_1.pt'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)

# After you've trained a well-performing model, answer the following questions so that we have some insight into your training and architecture selection process. Answering all questions is required to pass this project.

Question 1: What optimization and loss functions did you choose and why?

Answer: 

Question 2: What kind of network architecture did you start with and how did it change as you tried different architectures? Did you decide to add more convolutional layers or any layers to avoid overfitting the data?

Answer: 

Question 3: How did you decide on the number of epochs and batch_size to train your model?

Answer: 
	
# Feature Visualization
# Sometimes, neural networks are thought of as a black box, given some input, they learn to produce some output. CNN's are actually learning to recognize a variety of spatial patterns and you can visualize what each convolutional layer has been trained to recognize by looking at the weights that make up each convolutional kernel and applying those one at a time to a sample image. This technique is called feature visualization and it's useful for understanding the inner workings of a CNN.
# In the cell below, you can see how to extract a single filter (by index) from your first convolutional layer. The filter should appear as a grayscale grid.
# Get the weights in the first conv layer, "conv1"
# if necessary, change this to reflect the name of your first conv layer

weights1 = net.conv1.weight.data

w = weights1.numpy()

filter_index = 0

print(w[filter_index][0])
print(w[filter_index][0].shape)

# display the filter weights
plt.imshow(w[filter_index][0], cmap='gray')

# Feature maps
# Each CNN has at least one convolutional layer that is composed of stacked filters (also known as convolutional kernels). 
# As a CNN trains, it learns what weights to include in it's convolutional kernels and when these kernels are applied to some input image, 
# they produce a set of feature maps. So, feature maps are just sets of filtered images; they are the images produced by applying a convolutional 
# kernel to an input image. These maps show us the features that the different layers of the neural network learn to extract. For example, 
# you might imagine a convolutional kernel that detects the vertical edges of a face or another one that detects the corners of eyes. 
# You can see what kind of features each of these kernels detects by applying them to an image. One such example is shown below; 
# from the way it brings out the lines in an the image, you might characterize this as an edge detection filter.

# TODO: Filter an image to see the effect of a convolutional kernel

##TODO: load in and display any image from the transformed test dataset
## TODO: Using cv's filter2D function,
## apply a specific set of filter weights (like the one displayed above) to the test image

Question 4: Choose one filter from your trained CNN and apply it to a test image; what purpose do you think it plays? What kind of feature do you think it detects?

Answer: (does it detect vertical lines or does it blur out noise, etc.) write your answer here


# Moving on!
# Now that you've defined and trained your model (and saved the best model), you are ready to move on to the last notebook, 
# which combines a face detector with your saved model to create a facial keypoint detection system that can predict the keypoints on any face in an image!
                                                                       