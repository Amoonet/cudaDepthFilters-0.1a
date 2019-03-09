#ifndef CUDARADIUSOUTLIERREMOVAL_H
#define CUDARADIUSOUTLIERREMOVAL_H
#include <cuda.h>
#include <opencv2/opencv.hpp>

class cudaRadiusOutlierRemoval
{
private:
    float *depth;
    float *out;
    int rows;
    int cols;
public:
    cudaRadiusOutlierRemoval(int rows, int cols);
    ~cudaRadiusOutlierRemoval();

    void filter(cv::Mat &depth_mat, int window, int n_neighbours, bool use_intrinsic=false, float fx=1, float cx=1);
};

#endif // CUDARADIUSOUTLIERREMOVAL_H
