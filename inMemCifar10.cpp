#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace caffe;
using namespace std;
using namespace cv;
Mat frame;
int main(int argc, char** argv) {

    Caffe::set_mode(Caffe::GPU); // select CPU or GPU
    int deviceId=0;
    Caffe::SetDevice(deviceId);

    Net<float> net("examples/cifar10/cifar10_memory.prototxt",caffe::TEST); //get the net

    net.CopyTrainedLayersFrom( "examples/cifar10/cifar10_quick_iter_5000.caffemodel" );

    frame=imread("testImage.png",1);

    Timer total_timer;
    total_timer.Start();

    vector<cv::Mat> imageVector;
    imageVector.push_back(frame);

    vector<int> labelVector;
    labelVector.push_back(0);//push_back 0 for initialize purpose

    

    // Net initialization
    float loss = 0.0;
    boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;

    memory_data_layer.reset();
    memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net.layer_by_name("cifar"));
    memory_data_layer->AddMatVector(imageVector,labelVector);

    // Run ForwardPrefilled
    const vector<Blob<float>*>& results = net.ForwardPrefilled(&loss);

    printf("Time taken: %.5f ms\n", total_timer.MicroSeconds() /1000);

    // I really should read this from batches.meta.txt rather than hard code 
     string cifar10class[] = {"airplane", "automobile", "bird", "cat", "deer", 
                               "dog", "frog", "horse", "ship", "truck"};

    printf("\n");
    float maxP = 0;
    int idxMaxP = 0;
    for (int i = 0; i < 10; ++i) {
        float prediction = results[1]->cpu_data()[i];
        if (1 || prediction > 0.001){
           printf("%12s : %1.3f\n",cifar10class[i].c_str(),prediction);
        } else {
           printf("%12s : \n",cifar10class[i].c_str());
        }
        if (maxP < prediction){
            maxP = prediction;
            idxMaxP = i;
        }
    }

    cout << "Classified as: " << cifar10class[idxMaxP] << endl;

   
    return 0;
}

