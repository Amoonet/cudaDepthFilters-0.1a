#ifndef ENHANCEFROMNEIGHBOUR_H
#define ENHANCEFROMNEIGHBOUR_H
#include <cuda.h>
#include<cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include<math.h>
//#include<math_functions.h>
#include<time.h>

class enhanceFromNeighbour
{
private:
    int rows;
    int cols;

    float **cudaDepthBuffer;
    float **hostDepthBuffer;

    float** hostR, **hostT, **hostIntr;
    float** cudaR, **cudaT, **cudaIntr;
    unsigned int *cudaOut;
    int num_of_neighbours;

public:
    enhanceFromNeighbour(int rows, int cols, u_int num_of_devices);
    ~enhanceFromNeighbour();


    void filter(std::vector<cv::Mat> depths, cv::Mat &out, std::vector<cv::Mat> intrinsicParams, std::vector<cv::Mat> R, std::vector<cv::Mat> t);
};


#endif // ENHANCEFROMNEIGHBOUR_H
