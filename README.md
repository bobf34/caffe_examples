---
title: Caffe Examples
category: example
---

# Background

Because of change in caffe, many of the available examples on the web don't seem to work.   

Here you will find examples that, at least for a short while, should work.

# Examples 

## testCifar10.py

This example builds on the cifar10 example included with [caffe](https://github.com/BVLC/caffe).  You need to get this example working before using this script as it uses the training data.

The script reports the same accuracy as the cifar10 example.  It also generates a confusion matrix (reported in %), and plots the filters for the first convolution layer.  The confusion matrix shows the difficulty that the net had in distinguishing dog and cats.

'''
processed 1000
processed 2000
processed 3000
processed 4000
processed 5000
processed 6000
processed 7000
processed 8000
processed 9000
processed 10000
Total Accuracy: 82.0% 

['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
[ 83.6   1.4   3.4   1.4   1.    0.3   0.4   0.8   4.5   3.2] airplane
[  1.4  89.2   0.3   0.5   0.2   0.4   0.7   0.2   1.8   5.3] automobile
[  5.2   0.1  73.8   4.    5.5   3.2   4.7   1.9   0.5   1.1] bird
[  1.1   0.6   5.1  67.4   3.4  13.    4.2   2.8   1.1   1.3] cat
[  0.9   0.2   4.9   4.2  80.5   1.6   2.5   4.    1.    0.2] deer
[  1.    0.    3.8  13.2   3.3  73.9   0.8   3.5   0.4   0.1] dog
[  0.8   0.    4.    4.3   1.6   1.5  86.9   0.5   0.2   0.2] frog
[  1.    0.4   1.8   2.9   3.    4.1   0.6  85.6   0.1   0.5] horse
[  4.4   1.5   0.5   1.    0.6   0.4   0.2   0.3  89.1   2. ] ship
[  2.1   3.7   0.5   0.6   0.2   0.3   0.2   0.5   1.9  90. ] truck
'''

![alt text](https://github.com/bobf34/caffe_examples/blob/master/screenshots/caffeConv1Filters.png "conv1 filters")


When enabled, this program can also show images with correct and/or incorrect classifications as shown here.

![alt text](https://github.com/bobf34/caffe_examples/blob/master/screenshots/caffeWrongClass.png "Image with wrong classification")

 
