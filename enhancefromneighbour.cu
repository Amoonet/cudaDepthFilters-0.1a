#include "enhancefromneighbour.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__
void cudaEnhanceFromNeighbours(float **depths, unsigned int *out, float **intrinsicParams, float **R, float **t,
                              int rows, int cols,int num_of_devices){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int index = blockIdx.x;
    int arow = row/cols;
    int acol = row%cols;

    if (arow>=rows || acol>=cols) return;

    //from float to uint
    out[row] = depths[0][row]*((pow(2,16)-1));

    for(int k = 0; k<num_of_devices-1;k++){
        double p1 = depths[k+1][arow*cols + acol];
        if (p1 == 0)continue;

        //2D pixel to 3D point
        float v[3] = {p1*(acol+0.5-intrinsicParams[k+1][2])/intrinsicParams[k+1][0],
                      p1*(arow+0.5-intrinsicParams[k+1][3])/intrinsicParams[k+1][1],
                      p1};

        //translation
        v[0] -=t[k+1][0]/1000.0;
        v[1] -=t[k+1][1]/1000.0;
        v[2] -=t[k+1][2]/1000.0;

        //rotation
        float v2[3] = { ((v[0] * R[k+1][0]) + (v[1] * R[k+1][3]) + (v[2] * R[k+1][6])) ,
                        ((v[0] * R[k+1][1]) + (v[1] * R[k+1][4]) + (v[2] * R[k+1][7])) ,
                        ((v[0] * R[k+1][2]) + (v[1] * R[k+1][5]) + (v[2] * R[k+1][8])) };


        float x = round((v2[0]*intrinsicParams[0][0]/v2[2]) - 0.5 + intrinsicParams[0][2]);
        float y = round((v2[1]*intrinsicParams[0][1]/v2[2]) - 0.5 + intrinsicParams[0][3]);

        if((x<cols-1 && y<rows-1) && (x>=0 && y>=0)){
            atomicCAS(&out[(int)(y)*cols + (int)(x)],0,u_int(v2[2]*((pow(2,16)-1))));       //replace zeros with any values from other sensors
//          atomicMin(&out[(int)(y)*cols + (int)(x)], u_int(v2[2]*((pow(2,16)-1))));        //replace value with the smalest

        }
    }
}



/**************************************************************************************************
 * Allocation of memory
 * rows, cols     - size of input depth map
 * num_of_devices - number of devices used for enhancement
 * ************************************************************************************************/
enhanceFromNeighbour::enhanceFromNeighbour(int rows, int cols, u_int num_of_devices)
{
    hostDepthBuffer = (float**)malloc(sizeof(float*)*num_of_devices);
    hostR = (float**)malloc(sizeof(float*)*num_of_devices);
    hostT = (float**)malloc(sizeof(float*)*num_of_devices);
    hostIntr = (float**)malloc(sizeof(float*)*num_of_devices);

    for(int i = 0; i<num_of_devices; i++){
        gpuErrchk(  cudaMalloc((void**)&hostDepthBuffer[i], sizeof(float)*rows*cols)    );
        gpuErrchk(  cudaMalloc((void**)&hostR[i], sizeof(float)*9)    );
        gpuErrchk(  cudaMalloc((void**)&hostT[i], sizeof(float)*3)    );
        gpuErrchk(  cudaMalloc((void**)&hostIntr[i], sizeof(float)*4)    );
    }

    gpuErrchk(  cudaMalloc((void**)&cudaDepthBuffer, sizeof(float*) * num_of_devices)  );
    gpuErrchk(  cudaMalloc((void**)&cudaR, sizeof(float*) * num_of_devices)  );
    gpuErrchk(  cudaMalloc((void**)&cudaT, sizeof(float*) * num_of_devices)  );
    gpuErrchk(  cudaMalloc((void**)&cudaIntr, sizeof(float*) * num_of_devices)  );

    gpuErrchk(  cudaMalloc((void**)&cudaOut, sizeof(unsigned int)*rows*cols)    );
    this->rows = rows;
    this->cols = cols;
    num_of_neighbours = num_of_devices;

}

enhanceFromNeighbour::~enhanceFromNeighbour()
{
    for(int i = 0; i<num_of_neighbours; i++){
        gpuErrchk(  cudaFree(hostDepthBuffer[i])  );
        gpuErrchk(  cudaFree(hostR[i])  );
        gpuErrchk(  cudaFree(hostT[i])  );
        gpuErrchk(  cudaFree(hostIntr[i])  );
    }

    free(hostDepthBuffer);
    free(hostR);
    free(hostT);
    free(hostIntr);

    gpuErrchk(  cudaFree(cudaDepthBuffer)  );
    gpuErrchk(  cudaFree(cudaR)  );
    gpuErrchk(  cudaFree(cudaT)  );
    gpuErrchk(  cudaFree(cudaIntr)  );

}

/*******************************************************************************/
/*Inputs:
 *      depths      - vector of depth maps (types: CV_32F, CV_64F, CV_16U, CV_8U)
 *      intrinsics  - vector of intrinsics parameters: cv::Mat mat1 = (cv::Mat_<float>(1,4)<<fx, fy, cx, cy);
 *      R           - rotation matrices cv::Mat R1 = (cv::Mat_<float>(3,3)<<1,0,0,
 *                                                                      0,1,0,
 *                                                                      0,0,1);
 *      t           - translation vectors cv::Mat t1 = (cv::Mat_<float>(1,3)<<1,0,0);
 * output:
 *      out         - output depth map (type: same as inputs)
 ********************************************************************************/

void enhanceFromNeighbour::filter(std::vector<cv::Mat> depths, cv::Mat &out,std::vector<cv::Mat> intrinsicParams, std::vector<cv::Mat> R, std::vector<cv::Mat> t){
    assert(depths.size()>0);
    assert(!(depths.size()==intrinsicParams.size()==R.size()==t.size()));

    for(int i = 0; i < depths.size();i++){
        assert(depths.at(i).depth() == CV_32F || depths.at(i).depth() == CV_64F || depths.at(i).depth() == CV_16U || depths.at(i).depth() == CV_8U);
        assert(intrinsicParams.at(i).depth() == CV_32F);
    }

    out = cv::Mat::zeros(depths.at(0).size(),CV_32S);

    int type =  depths.at(0).depth();
    for(int i = 0; i < depths.size();i++){
        if(depths.at(i).depth() == CV_16U){
            depths.at(i).convertTo(depths.at(i),CV_32F,1.0/(pow(2,16)-1),0);
        }
        else if(depths.at(i).depth() == CV_8U){
            depths.at(i).convertTo(depths.at(i),CV_32F,1.0/(pow(2,8)-1),0);
        }
        else if(depths.at(i).depth() == CV_64F){
            depths.at(i).convertTo(depths.at(i),CV_32F);
        }
    }

    for (int i = 0; i < depths.size(); i++){
        gpuErrchk(  cudaMemcpy(hostDepthBuffer[i], depths.at(i).data, sizeof(float)*this->rows*this->cols, cudaMemcpyHostToDevice)    );
        gpuErrchk(  cudaMemcpy(hostR[i], R.at(i).data, sizeof(float)*9, cudaMemcpyHostToDevice)   );
        gpuErrchk(  cudaMemcpy(hostT[i], t.at(i).data, sizeof(float)*3, cudaMemcpyHostToDevice)   );
        gpuErrchk(  cudaMemcpy(hostIntr[i], intrinsicParams.at(i).data, sizeof(float)*4, cudaMemcpyHostToDevice)  );
    }

    gpuErrchk(  cudaMemcpy(cudaDepthBuffer, hostDepthBuffer, sizeof(float*) * depths.size(), cudaMemcpyHostToDevice)   );
    gpuErrchk(  cudaMemcpy(cudaR, hostR, sizeof(float*) * depths.size(), cudaMemcpyHostToDevice)     );
    gpuErrchk(  cudaMemcpy(cudaT, hostT, sizeof(float*) * depths.size(), cudaMemcpyHostToDevice)     );
    gpuErrchk(  cudaMemcpy(cudaIntr, hostIntr, sizeof(float*) * depths.size(), cudaMemcpyHostToDevice)   );

    u_int N = this->rows*this->cols;
    u_int BLOCK_SIZE = N<1024?N:1024;
    dim3 dimBlock = dim3(BLOCK_SIZE,1);
    dim3 dimGrid = dim3(N/BLOCK_SIZE + 1,1);

    cudaEnhanceFromNeighbours<<< dimGrid,dimBlock>>>(cudaDepthBuffer,cudaOut,cudaIntr,cudaR,cudaT,this->rows,this->cols,depths.size());
    cudaDeviceSynchronize();
    gpuErrchk(  cudaMemcpy((unsigned int*)out.data,cudaOut, sizeof(unsigned int)*this->rows*this->cols, cudaMemcpyDeviceToHost)    );

    if(type == CV_32F){
        out.convertTo(out,CV_32F,1.0/(pow(2,16)-1));
    }
    else if(type == CV_8U){
        out.convertTo(out,CV_8U,(255.0/pow(2,8)-1),0);
    }
    else if(type == CV_64F){
        out.convertTo(out,CV_64F);
    }
}
