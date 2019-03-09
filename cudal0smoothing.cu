#include "cudal0smoothing.h"

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
void cudaL0SmoothP1(double* depth,u_char *ir,double** h, double** v,float beta,float lambda,int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int arow = row/cols;
    int acol = row%cols;
    if (arow>=rows || acol>=cols) return;
    float w1 = 1.0, w2 = 1;
    double h2 = 0,v2 = 0;
    if(acol<cols-1){
        h[0][row] = (depth[row+1]-depth[row]) * w1;
        h2 = ((ir[row+1]-ir[row])/255.0) * w2;
    }
    else{
        h[0][row] = (depth[(row/cols)*cols]-depth[row]) * w1;
        h2 = ((ir[(row/cols)*cols]-ir[row])/255.0) * w2;
    }

    if(arow<rows-1){
        v[0][row] = (depth[row+cols]-depth[row]) * w1;
        v2 = ((ir[row+cols]-ir[row])/255.0) * w2 ;
    }
    else{
        v[0][row] = (depth[acol]-depth[row]) * w1;
        v2 = ((ir[acol]-ir[row])/255.0) * w2 ;
    }

    h[0][row] = (h[0][row]*h[0][row]) + (v[0][row]*v[0][row])<lambda/beta && (h2*h2) + (v2*v2)<lambda/beta?0:h[0][row];
    v[0][row] = (h[0][row]*h[0][row]) + (v[0][row]*v[0][row])<lambda/beta && (h2*h2) + (v2*v2)<lambda/beta?0:v[0][row];
}

__global__
void cudaL0SmoothP2(double** h,double** v, double* normin2,float beta,float lambda,int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int arow = row/cols;
    int acol = row%cols;
    if (arow>=rows || acol>=cols) return;


    if(acol>0)
        normin2[row] = -(h[0][row]-h[0][row-1]);
    else{
        normin2[row] = h[0][(row/cols)*cols + (cols-1)]-h[0][row];
    }

    if(arow>0)
        normin2[row] += -(v[0][row]-v[0][row-cols]);
    else{
        normin2[row] += v[0][(rows-1)*cols + acol]-v[0][row];
    }
}

__global__
void cudaL0SmoothP3(cufftDoubleComplex* normin1, cufftDoubleComplex* normin2, cufftDoubleComplex* FS,double *absValues,float beta,int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int arow = row/cols;
    int acol = row%cols;
    if (arow>=rows || acol>=cols) return;

    double denormin = 1+(beta*absValues[row]);
    FS[row].x = ((normin1[row].x + beta*normin2[row].x)/denormin);
    FS[row].y = ((normin1[row].y + beta*normin2[row].y)/denormin);
}

__global__
void scale(double*indata, double scaleFactor,int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int arow = row/cols;
    int acol = row%cols;
    if (arow>=rows || acol>=cols){
        return;
    }
    indata[row] = indata[row]/scaleFactor;
}

__global__
void cudaAbsValues(cufftDoubleComplex * data1,cufftDoubleComplex * data2, double* val,int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int arow = row/cols;
    int acol = row%cols;
    if (arow>=rows || acol>=cols) return;
    float D1 = 0, D2 = 0;

    D1 = sqrt(pow(data1[row].x,2) + pow(data1[row].y,2));
    D2 = sqrt(pow(data2[row].x,2) + pow(data2[row].y,2));

    val[row] = pow(D1,2) + pow(D2,2);
}

/**************************************************************************************************
 * Allocation of memory
 * rows, cols       - size of input depth map
 * ************************************************************************************************/
cudaL0Smoothing::cudaL0Smoothing(int rows, int cols){
    //allocating necessary memory
    gpuErrchk(  cudaMalloc((void **) &depth, rows*cols*sizeof(double))     );
    gpuErrchk(  cudaMalloc((void **) &ir, rows*cols*sizeof(uchar))     );
    gpuErrchk(  cudaMalloc((void **) &cudaOutReal, rows*cols*sizeof(cufftDoubleReal))     );
    gpuErrchk(  cudaMalloc((void **) &cudaOutComplex, rows*cols*sizeof(cufftDoubleComplex))     );
    gpuErrchk(  cudaMalloc((void **) &cudaOutComplex2, rows*cols*sizeof(cufftDoubleComplex))     );
    gpuErrchk(  cudaMalloc((void **) &cudaFS, rows*cols*sizeof(cufftDoubleComplex))     );
    gpuErrchk(  cudaMalloc((void **) &normin2, rows*cols*sizeof(double))     );

    this->rows = rows;
    this->cols = cols;

    hostH = (double**)malloc(sizeof(double*)*2);
    hostV = (double**)malloc(sizeof(double*)*2);

    for(int i = 0; i<2; i++){
        gpuErrchk(  cudaMalloc((void**)&hostH[i], sizeof(double)*rows*cols)    );
        gpuErrchk(  cudaMalloc((void**)&hostV[i], sizeof(double)*rows*cols)    );
    }

    gpuErrchk(  cudaMalloc((void**)&cudaH, sizeof(double*) * 2)  );
    gpuErrchk(  cudaMalloc((void**)&cudaV, sizeof(double*) * 2)  );
    gpuErrchk(  cudaMemcpy(cudaH, hostH, sizeof(double*) * 2, cudaMemcpyHostToDevice)  );
    gpuErrchk(  cudaMemcpy(cudaV, hostV, sizeof(double*) * 2, cudaMemcpyHostToDevice)  );
    gpuErrchk(  cudaMalloc((void**)&absValues, sizeof(double)*rows*cols)    );


    cufftDoubleComplex *d_Kx, *d_Ky;
    double *Kx,*Ky;
    gpuErrchk(  cudaMalloc((void**)&d_Kx, sizeof(cufftDoubleComplex)*rows*cols)  );
    gpuErrchk(  cudaMalloc((void**)&d_Ky, sizeof(cufftDoubleComplex)*rows*cols)  );
    gpuErrchk(  cudaMalloc((void**)&Kx, sizeof(double)*rows*cols)  );
    gpuErrchk(  cudaMalloc((void**)&Ky, sizeof(double)*rows*cols)  );

    cv::Mat Kfx = cv::Mat::zeros(rows,cols,CV_64F);
    cv::Mat Kfy = cv::Mat::zeros(rows,cols,CV_64F);

    //creating KERNEL
    Kfx.at<double>(0,0) = -1;
    Kfx.at<double>(0,cols-1) = 1;
    Kfy.at<double>(0,0) = -1;
    Kfy.at<double>(rows-1,0) = 1;

    gpuErrchk(  cudaMemcpy(Kx, Kfx.data ,rows*cols*sizeof(double),cudaMemcpyHostToDevice)     );
    gpuErrchk(  cudaMemcpy(Ky, Kfy.data ,rows*cols*sizeof(double),cudaMemcpyHostToDevice)     );

    //2DFFT
    cufftDoubleComplex *comp = FFT2D_Z2C(rows,cols,Kx);
    gpuErrchk(  cudaMemcpy(d_Kx, comp ,rows*cols*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice)     );
    comp = FFT2D_Z2C(rows,cols,Ky);
    gpuErrchk(  cudaMemcpy(d_Ky, comp ,rows*cols*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice)     );

    int N = rows*cols;
    int BLOCK_SIZE = N<1024?N:1024;
    dim3 dimBlock(BLOCK_SIZE,1);
    dim3 dimGrid(N/BLOCK_SIZE + 1,1);

    //get absolute values
    cudaAbsValues<<<dimGrid,dimBlock>>>(d_Kx,d_Ky, absValues,rows,cols);

    cudaFree(d_Kx);
    cudaFree(d_Ky);
    cudaFree(Kx);
    cudaFree(Ky);

}
cudaL0Smoothing::~cudaL0Smoothing(){
    gpuErrchk(  cudaFree(depth)  );
    gpuErrchk(  cudaFree(ir)  );
    gpuErrchk(  cudaFree(cudaOutReal)  );
    gpuErrchk(  cudaFree(cudaOutComplex)  );
    gpuErrchk(  cudaFree(cudaOutComplex2)  );
    gpuErrchk(  cudaFree(cudaFS)  );

    gpuErrchk(  cudaFree(normin2)  );

    for(int i = 0; i<2; i++){
        gpuErrchk(  cudaFree(hostH[i])  );
        gpuErrchk(  cudaFree(hostV[i])  );
    }
    free(hostH);
    free(hostV);
    gpuErrchk(  cudaFree(cudaH)  );
    gpuErrchk(  cudaFree(cudaV)  );
    gpuErrchk(  cudaFree(absValues)  );


}

/*******************************************************************************/
/*Inputs:
 *      depths_mat      - depth map (types: CV_32F, CV_64F, CV_16U, CV_8U)
 *      ir_mat          - weight map, usually ir image or grayscale image
 *      kappa           - kappa parameter
 *      lambda          - lambda parameter
 ********************************************************************************/
void cudaL0Smoothing::filter(cv::Mat &depth_mat,cv::Mat ir_mat,float kappa,float lambda){

    assert(depth_mat.depth() == CV_32F || depth_mat.depth() == CV_64F || depth_mat.depth() == CV_16U || depth_mat.depth() == CV_8U);
    assert(ir_mat.depth() == CV_8U && ir_mat.channels()==1);

    int type =  depth_mat.depth();
    if(depth_mat.depth() == CV_16U){
        depth_mat.convertTo(depth_mat,CV_64F,1.0/(pow(2,16)-1),0);
    }
    else if(depth_mat.depth() == CV_8U){
        depth_mat.convertTo(depth_mat,CV_64F,1.0/(pow(2,8)-1),0);
    }
    else if(depth_mat.depth() == CV_32F){
        depth_mat.convertTo(depth_mat,CV_64F);
    }

    gpuErrchk(  cudaMemcpy(depth, (double*)depth_mat.data, sizeof(double) * depth_mat.rows * depth_mat.cols, cudaMemcpyHostToDevice)  );
    gpuErrchk(  cudaMemcpy(ir, (double*)ir_mat.data, sizeof(uchar) * ir_mat.rows * ir_mat.cols, cudaMemcpyHostToDevice)  );

    for(int i =0;i<2;i++){
        gpuErrchk(  cudaMemset(hostH[i], 0, sizeof(double)*rows*cols)  );
        gpuErrchk(  cudaMemset(hostV[i], 0, sizeof(double)*rows*cols)  );
    }

    gpuErrchk(  cudaMemcpy(cudaH, hostH, sizeof(double*) * 2, cudaMemcpyHostToDevice)  );
    gpuErrchk(  cudaMemcpy(cudaV, hostV, sizeof(double*) * 2, cudaMemcpyHostToDevice)  );

    u_int N = this->rows*this->cols;
    u_int BLOCK_SIZE = N<1024?N:1024;
    dim3 dimBlock = dim3(BLOCK_SIZE,1);
    dim3 dimGrid = dim3(N/BLOCK_SIZE + 1,1);

    cufftDoubleComplex *comp =  FFT2D_Z2C(this->rows,this->cols,depth);
    gpuErrchk(  cudaMemcpy(cudaOutComplex2, comp ,rows*cols*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice)     );

    float beta = 2*lambda;
    float betamax = 1e5;

    //core
    while(beta<betamax){
//        printf("-------------------------------------\n");
        cudaL0SmoothP1<<<dimGrid,dimBlock>>>(depth,ir,cudaH,cudaV,beta,lambda,this->rows,this->cols);
        cudaDeviceSynchronize();

        cudaL0SmoothP2<<<dimGrid,dimBlock>>>(cudaH,cudaV,normin2,beta,lambda,this->rows,this->cols);
        cudaDeviceSynchronize();

        //2DFFT
        comp = FFT2D_Z2C(this->rows,this->cols,normin2);

        cudaL0SmoothP3<<<dimGrid,dimBlock>>>(cudaOutComplex2,comp, cudaFS,absValues,beta,rows,cols);
        cudaDeviceSynchronize();

        //2DIFFT
        cufftDoubleReal *real = FFT2D_C2Z(this->rows,this->cols,cudaFS);

        scale<<<dimGrid,dimBlock>>>(real, this->rows*this->cols,rows,cols);
        cudaDeviceSynchronize();
        gpuErrchk(  cudaMemcpy(depth, real,this->rows*this->cols*sizeof(double),cudaMemcpyDeviceToDevice) );
        beta *=kappa;
    }
    gpuErrchk(  cudaMemcpy((double*)depth_mat.data, depth,this->rows*this->cols*sizeof(double),cudaMemcpyDeviceToHost) );

    if(type == CV_16U){
        depth_mat.convertTo(depth_mat,CV_16U,pow(2,16)-1,0);
    }
    else if(type == CV_8U){
        depth_mat.convertTo(depth_mat,CV_8U,pow(2,8)-1,0);
    }
    else if(type == CV_32F){
        depth_mat.convertTo(depth_mat,CV_32F);
    }
}

cufftDoubleComplex*  cudaL0Smoothing::FFT2D_Z2C( const unsigned int Nx, const unsigned int Ny, cufftDoubleReal *data){

    gpuErrchk(  cudaMemset(cudaOutComplex, 0, sizeof(cufftDoubleComplex)*Nx*Ny)    );

    cufftPlan2d(&plan, Nx, Ny, CUFFT_D2Z );

    flag = cufftExecD2Z( plan, data, cudaOutComplex );

    if ( CUFFT_SUCCESS != flag ){
        printf("2D: cufftExecR2C fails\n");
    }
    cudaThreadSynchronize();
    cufftDestroy(plan);
    return cudaOutComplex;

}

cufftDoubleReal*  cudaL0Smoothing::FFT2D_C2Z( const unsigned int Nx, const unsigned int Ny, cufftDoubleComplex *data){

    gpuErrchk(  cudaMemset(cudaOutReal, 0, sizeof(cufftDoubleReal)*Nx*Ny)    );

//    gpuErrchk(  cudaMemcpy(d_idata,h_idata,sizeof(cufftDoubleComplex)*Nx*Ny,cudaMemcpyHostToDevice)  );


    cufftPlan2d(&plan, Nx, Ny, CUFFT_Z2D);
    flag = cufftExecZ2D( plan, data, cudaOutReal);

    if ( CUFFT_SUCCESS != flag ){
        printf("2D: cufftExecR2C fails\n");
    }

    cudaThreadSynchronize();

    cufftDestroy(plan);
    return cudaOutReal;
}
