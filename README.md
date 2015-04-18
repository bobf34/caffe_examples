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

The script reports the same accuracy as the cifar10 example.  It also generates a confusion matrix, and plots the filters for the first convolution layer.

![alt text](https://github.com/bobf34/caffe_examples/blob/master/screenshots/caffeConv1Filters.png "conv1 filters")


When enabled, this program can also show images with correct and/or incorrect classifications as shown here.

![alt text](https://github.com/bobf34/caffe_examples/blob/master/screenshots/caffeWrongClass.png "Image with wrong classification")

 
