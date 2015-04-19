---
title: Caffe Examples
category: example
---

# Background

Because of ongoing development in [caffe](https://github.com/BVLC/caffe), it seems that many of the available examples on the web no longer work; some never did. Working, or not, these examples have provided some useful clues as to how to use caffe along with a number of dead-ends!

The two examples here leverage, one in python, the other in c++, combine, repair and extend the work of others. They are compatible with the master branch of caffe (as available April 2015).  I hope they will save you both time and frustration, and when they break, which I'm sure they will, that they'll at least serve as a newer broken examples.

_Bob_

# Examples 

## testCifar10.py

This example builds on the cifar10 example included with [caffe](https://github.com/BVLC/caffe).  You need to get this example working before using this script as it uses the training data.

**Note:**  The script is set up to use a GPU, you'll need to edit the script if you haven't compiled caffe with GPU support, or you don't have a GPU.

This python script may help you if:
* You want to feed caffe an image in memory 
* You want to read data from an lmdb database
* You want to view the network filters
* You want plots from caffe that look similar to the ones in published papers


The script reports the same accuracies as the cifar10 examples (e.g. train_full.sh).  It also generates a confusion matrix, reported in percent, After running a full test, it then also plots the filters for the first convolution layer.  

The confusion matrix indicates that the net had in distinguishing dog and cats.  If you haven't looked at a confusion matrix before, the table below indicates that 5.2% of the bird images were classified as airplanes that 3.4% of the airplanes were classified as birds. (Sorry about the label formating across the top!)

```
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
```

![alt text](https://github.com/bobf34/caffe_examples/blob/master/screenshots/caffeConv1Filters.png "conv1 filters")


When enabled, this program can also show images with correct and/or incorrect classifications as shown here.  The correct and incorrect classes are show at the top, and the top 5 classes are displayed as in bargraph form at the bottom.

![alt text](https://github.com/bobf34/caffe_examples/blob/master/screenshots/caffeWrongClass.png "Image with wrong classification")

##inMemCifar10.cpp

This program reads a single 32x32 pixel image from a file called testImage.png. From memory, it is then passed to the data layer (type: "MemoryData), where it is then processed using the cifar10 quick training data.   The MemoryData layer wasn't used in the cifar10 example, so a new prototxt file is inlcuded.  The cifar10_memory.prototxt file should be placed in the examples/cifar10/ directory.

**Note:**  The program is set up to use a GPU, you'll need to edit the program if you haven't compiled caffe with GPU support, or you don't have a GPU.  The 2ms run time shown below was made using a GTX 780.  It's about 5x faster than runnin on my CPU.

The easiest way to compile and run the program is to put the cpp file into the caffe/tools directory and place the testImage.png file (included) in the caffe root directory.  Then, from the caffe root directory, type 'make'.  To run the program type './build/tools/inMemCifar10'

If you used my testImage.png file ![alt text](testImage.png "32x32 pixel image of a boat"), you should see something like this in the console output indicating a 94% probability of this being an image of a ship.

```
Time taken: 2.22208 ms

    airplane : 0.000
  automobile : 0.034
        bird : 0.000
         cat : 0.000
        deer : 0.000
         dog : 0.000
        frog : 0.000
       horse : 0.000
        ship : 0.941
       truck : 0.024
Classified as: ship

```




