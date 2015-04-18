# Place this file in the root caffe directory select options below

# Make sure that you've trained the net according to the cifar10 instructions
# included with the cifar10 example

# Set the options below

# Set plotErrors to True to view images which fail  
#    Setting to True will cause the cause the image to be shown along
#    with a bar graph
plotErrors = False
plotAll = False

#choose which results to test
useQuickTrain = False


import numpy as np
import lmdb
import sys
import time
import pdb
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, './python') # just to import caffe python library

import caffe
import caffe.proto
import caffe.io

from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array


if useQuickTrain:  
     # shoul give 75.1% accuarcy
     MODEL_FILE = 'examples/cifar10/cifar10_quick.prototxt'
     PRETRAINED = 'examples/cifar10/cifar10_quick_iter_5000.caffemodel'
else:
     # shoul give 82% accuarcy
     MODEL_FILE = 'examples/cifar10/cifar10_full.prototxt'
     PRETRAINED = 'examples/cifar10/cifar10_full_iter_70000.caffemodel'


MEAN_FILE= 'examples/cifar10/mean.binaryproto'
META_FILE = 'data/cifar10/batches.meta.txt'

def load_labels(meta_file=META_FILE):
    with open(meta_file) as mfile:
        labelNames = [x.strip('\n') for x in mfile.readlines()]
    labelNames.remove('')
    return labelNames

def load_mean(mean_file=MEAN_FILE):
    blob = caffe_pb2.BlobProto()
    data = open(mean_file, "rb").read()
    blob.ParseFromString(data)
    nparray = blobproto_to_array(blob)
    return nparray[0] 

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

caffe.set_mode_gpu()

net = caffe.Net(MODEL_FILE, 
                PRETRAINED,
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,1,0))
#transformer.set_mean('data', load_mean()) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# read image from database
lmdb_env = lmdb.open('examples/cifar10/cifar10_test_lmdb')
#lmdb_env = lmdb.open('examples/cifar10/cifar10_train_lmdb')

lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

meanImage = load_mean()

count = 0
labels_set = set()
lbl_names = load_labels()
numLabels = len(lbl_names)

if plotErrors or plotAll:
    plt.subplot(211)
    plt.show(block=False)

confusionMatrix = np.zeros((numLabels,numLabels))
for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)
    normImg = image-meanImage
    #pdb.set_trace()

    #out = net.forward_all(data=np.asarray([image]))
    out = net.forward_all(data=np.asarray([normImg]))
    prediction = int(out['prob'][0].argmax(axis=0))

    confusionMatrix[label,prediction] += 1
    if confusionMatrix.sum() % 1000 == 0:
       print "processed %i" % confusionMatrix.sum()

    if (plotErrors and prediction != label) or plotAll:
        #pdb.set_trace()
        print lbl_names[label],' classified as ',lbl_names[prediction],' prob ',out['prob'][0][prediction] 

        #show the image
        plt.clf() 
        plt.subplot(211)
        plt.imshow(np.rollaxis(image,0,3))
        plt.title(lbl_names[label]+'-as-'+lbl_names[prediction])
        plt.subplot(212)
        #plot the top 5
        pos = np.arange(5)+.5
        probs,lbls = zip(*sorted(zip(out['prob'][0],lbl_names)))
        plt.barh(pos,probs[-5:], align='center')
        plt.yticks(pos,lbls[-5:])
        plt.axis([0, 1, 0, 5])
        plt.draw()
        time.sleep(5)


#normalize to percent.  The next few lines could be replaced by a simple divide by 10 for CIFAR
numEachClass = confusionMatrix.sum(1)  #should b 1000 in CIFAR
sumCorrect = 0
for i in range(confusionMatrix.shape[0]):
  confusionMatrix[i,:] *= 100 / numEachClass[i] 
  sumCorrect += confusionMatrix[i,i]

print "Total Accuracy: %.1f%% \n" % (sumCorrect/confusionMatrix.shape[0])
print lbl_names
for i in range(confusionMatrix.shape[0]):
    print confusionMatrix[i,:], lbl_names[i]

# dimentions of filters
[(k,(v[0].data.shape, v[1].data.shape)) for k, v in net.params.items()]

plt.figure(2)
vis_square(np.rollaxis(net.params['conv1'][0].data,1,4))
plt.show()
#pdb.set_trace()
