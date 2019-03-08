#ifndef CUDAcudaL0Smoothing_H
#define CUDAcudaL0Smoothing_H
#include <cuda.h>
#include <cufft.h>
#include <iostream>
#include <opencv2/opencv.hpp>

class cudaL0Smoothing{
public:
    void filter(cv::Mat &depth_mat,cv::Mat ir_mat,float kappa,float lambda);

    double *absValues;
    cudaL0Smoothing(int rows, int cols);
    ~cudaL0Smoothing();

private:

    cufftDoubleComplex* FFT2D_Z2C( const unsigned int Nx, const unsigned int Ny, cufftDoubleReal *h_idata);
    cufftDoubleReal* FFT2D_C2Z( const unsigned int Nx, const unsigned int Ny, cufftDoubleComplex *di_idata);

    cufftResult flag;
    cufftHandle plan;
    double **cudaH,**hostH,**cudadH;
    double **cudaV,**hostV,**cudadV;
    double *normin2;
    double *depth;
    uchar *ir;
    cufftDoubleReal *cudaOutReal;
    cufftDoubleComplex *cudaOutComplex;
    cufftDoubleComplex *cudaOutComplex2;
    cufftDoubleComplex* cudaFS;

    int rows;
    int cols;
};

#endif // CUDAcudaL0Smoothing_H
