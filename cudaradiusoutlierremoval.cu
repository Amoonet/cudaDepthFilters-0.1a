#include "cudaradiusoutlierremoval.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//difference: radius = abs((cuda_in[row]*(0.5-cx)/fx)-(cuda_in[row]*((window*2 + 1)+0.5-cx)/fx));
__global__
void cudaROR(float* cuda_in, float* cuda_out, int window, int numOfNeighbours,int cols, int rows, float fx, float cx){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int sizeofData = rows * cols;

    if (row > sizeofData) return;
    if(cuda_in[row]<0.0061){
        cuda_out[row] = 0;
        return;
    }

    int actual_grid_col = row%cols;
    int actual_grid_row = row/cols;

    //radius is calculated based on depth of point
    double radius = abs((cuda_in[row]*(0.5-cx)/fx)-(cuda_in[row]*((window*2 + 1)+0.5-cx)/fx));

    //find boundaries
    int start_x = actual_grid_row-window<0?0:actual_grid_row-window,
        start_y = actual_grid_col-window<0?0:actual_grid_col-window;
    int des_x = actual_grid_row+window>rows?rows:actual_grid_row+window,
        des_y = actual_grid_col+window>cols?cols:actual_grid_col+window;
    int counter = 0;

    //count number of neigbours
    for(int i=start_x;i<=des_x;i++){
        for(int j=start_y;j<=des_y;j++){
            if(cuda_in[i*cols+j]!=0){
                double _diff = (cuda_in[i*cols+j] - cuda_in[row]);
                double mag = _diff*_diff;
                if(mag<radius*radius){
                    counter++;
                }
            }
        }
    }

    if(counter<numOfNeighbours+1){
        cuda_out[row] = 0;
    }
    else{
        cuda_out[row] = cuda_in[row];
    }
}

//difference: radius = window
__global__
void cudaROR(float* cuda_in, float* cuda_out, int window, int numOfNeighbours,int cols, int rows){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int sizeofData = rows * cols;

    if (row > sizeofData) return;
    if(cuda_in[row]<0.0061){
        cuda_out[row] = 0;
        return;
    }

    int actual_grid_col = row%cols;
    int actual_grid_row = row/cols;

    float radius = window;

    //find boundaries
    int start_x = actual_grid_row-window<0?0:actual_grid_row-window,
        start_y = actual_grid_col-window<0?0:actual_grid_col-window;
    int des_x = actual_grid_row+window>rows?rows:actual_grid_row+window,
        des_y = actual_grid_col+window>cols?cols:actual_grid_col+window;
    int counter = 0;

    //count number of neigbours
    for(int i=start_x;i<=des_x;i++){
        for(int j=start_y;j<=des_y;j++){
            if(cuda_in[i*cols+j]!=0){
                double _diff = (cuda_in[i*cols+j] - cuda_in[row]);
                double mag = _diff*_diff;
                if(mag<radius*radius){
                    counter++;
                }
            }
        }
    }

    if(counter<numOfNeighbours+1){
        cuda_out[row] = 0;
    }
    else{
        cuda_out[row] = cuda_in[row];
    }
}

/**************************************************************************************************
 * Allocation of memory
 * rows, cols     - size of input depth map
 * ************************************************************************************************/
cudaRadiusOutlierRemoval::cudaRadiusOutlierRemoval(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;

    gpuErrchk(  cudaMalloc((void **) &depth, rows*cols*sizeof(double))     );
    gpuErrchk(  cudaMalloc((void **) &out, rows*cols*sizeof(double))     );
}

cudaRadiusOutlierRemoval::~cudaRadiusOutlierRemoval()
{
    gpuErrchk(  cudaFree(depth)  );
    gpuErrchk(  cudaFree(out)  );
}

/*******************************************************************************/
/*Inputs:
 *      depths_mat      - depth map (types: CV_32F, CV_64F, CV_16U, CV_8U)
 *      window          - size of window in pixels: radius = (window*2 + 1)
 *      n_neighbours    - limit of neigbours for depth pixel
 *      use_intrinsics  - use intrinsic paremeters fx and cx to calculate radius based in the depth
 ********************************************************************************/
void cudaRadiusOutlierRemoval::filter(cv::Mat &depth_mat, int window, int n_neighbours, bool use_intrinsic,float fx, float cx){

    assert(depth_mat.depth() == CV_32F || depth_mat.depth() == CV_64F || depth_mat.depth() == CV_16U || depth_mat.depth() == CV_8U);

    int type =  depth_mat.depth();
    if(depth_mat.depth() == CV_16U){
        depth_mat.convertTo(depth_mat,CV_64F,1.0/(pow(2,16)-1),0);
    }
    else if(depth_mat.depth() == CV_8U){
        depth_mat.convertTo(depth_mat,CV_64F,1.0/(pow(2,8)-1),0);
    }
    else if(depth_mat.depth() == CV_64F){
        depth_mat.convertTo(depth_mat,CV_32F);
    }

    gpuErrchk(  cudaMemcpy(depth, (float*)depth_mat.data ,this->rows*this->cols*sizeof(float),cudaMemcpyHostToDevice)     );

    int N = this->rows*this->cols;
    int BLOCK_SIZE = N<1024?N:1024;
    dim3 dimBlock(BLOCK_SIZE,1);
    dim3 dimGrid(N/BLOCK_SIZE + 1,1);

    //core
    if(use_intrinsic){
        cudaROR<<< dimGrid,dimBlock >>>(depth,out,window,n_neighbours,cols,rows,fx,cx);
        cudaDeviceSynchronize();
    }
    else{
        cudaROR<<< dimGrid,dimBlock >>>(depth,out,window,n_neighbours,cols,rows);
        cudaDeviceSynchronize();
    }
    gpuErrchk(  cudaMemcpy((float*)depth_mat.data,  out ,this->rows*this->cols*sizeof(float),cudaMemcpyDeviceToHost) );

    if(type == CV_16U){
        depth_mat.convertTo(depth_mat,CV_16U,pow(2,16)-1,0);
    }
    else if(type == CV_8U){
        depth_mat.convertTo(depth_mat,CV_8U,pow(2,8)-1,0);
    }
    else if(type == CV_64F){
        depth_mat.convertTo(depth_mat,CV_64F);
    }
}
