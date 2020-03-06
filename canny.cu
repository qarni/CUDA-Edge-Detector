#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil.h>
#include "canny.h"

// kernel function definitions

// device function defintions 
__device__ int getPixelVal(int* image, int height, int width, int x, int y);
__global__ void GaussianBlur(int* input, int* output, int height, int width, float* gaussianFilter, int kernelSize, int32_t* count);


// ------------------------------------------------------------------------------------

/*
Wrapper function to make kernel calls to perform canny algorithm 
*/
void canny(int* input, int height, int width, int* output, int kernelSize,  int sigma) {

    int matrixSize = height * width * sizeof(int);
    float* filter  = generateGaussianFilter(kernelSize, sigma);

    // set up for kernel calls 
    int* inputD = AllocateDeviceMemory(input, matrixSize);
    int* outputD = AllocateDeviceMemory(output, matrixSize);
    float* filterD;
    cudaMalloc(&filterD, kernelSize * kernelSize * sizeof(float));
    int32_t count = 0;
    int32_t* countD;
    cudaMalloc(&countD, sizeof(int32_t));
    
    CopyToDevice(&(input[0]), inputD, matrixSize);
    CopyToDevice(&(output[0]), outputD, matrixSize);
    CopyToDevice(&filter, filterD, kernelSize * kernelSize * sizeof(float));
    CopyToDevice(&count, countD, sizeof(int32_t));

    printf("input before: %d\5n",sizeof(input));

    for (int i = 0; i < height; i++)
    {
       for (int j = 0; j < width; j++)
       {
           output[width * i + j]= -2;
        //    printf("row: %d, col %d, val: %d address: %lx\n", i, j, input[width * i + j], input + width * i + j);
       }
    }

    // 2400 = 8  * 300
    // 600  = 8 * 75
    // 4 = 1  * 4

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(300, 300);

    GaussianBlur<<<numBlocks,threadsPerBlock>>>(inputD, outputD, height, width, filterD, kernelSize, countD);

    cudaThreadSynchronize();

    // tear down after kernel calls are done
    CopyFromDevice(outputD, &(output[0]), matrixSize);
    CopyFromDevice(countD, &count, sizeof(int32_t));
    cudaFree(inputD);
    cudaFree(outputD);
    cudaFree(filterD);
    cudaFree(countD);

    printf("\n\ndone with canny algorithm\n");
    printf("count: %d\n", count);
}

/*
This is the guassian filter to be applied over each pixel in the image


G(x, y) = (1/2*pi*sigma^2)*e^-(x^2+y^2/2*sigma^2)
*/
float*  generateGaussianFilter(int kernelSize, int sigma) {

    float* filter  = (float*) malloc(kernelSize * kernelSize * sizeof(float));

    float div = 2.0 * sigma *  sigma;
    float pre = 1.0 / (M_PI * div);

    int i  = 0;
    
    for (int x = -2; x <= 2; x++) { 
        for (int y = -2; y <= 2; y++) { 
            filter[i] = pre * pow(M_E, -((pow(x,2) + pow(y, 2)) / div));
            i++;
        }
    }

    return filter;
}

// kernel functions --------------------------------------------------------------------

/*
Noise reduction - gets rid of background noise but still keeps borders more in focus so they 
can be detected in the next step

Apply gaussian filter over each pixel 
Start with kernel size of 5?


*/
__global__ void GaussianBlur(int* input, int* output, int height, int width, float* gaussianFilter, int kernelSize, int32_t* count) {

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    int val = getPixelVal(input, height, width, row, col);
    if(val == -1)
        return;
    else  {
        // printf("row: %d, col %d, val: %d address: %lx\n", row, col, val, input + (width * row + col));
    }

    output[width * row + col] = val;

    __syncthreads();

    atomicAdd(count, 1);

    // __syncthreads();

}


// device functions --------------------------------------------------------------------
// can only be called from global func or from another device func, not from host

/*
returns pixel value at a location 
Maps a 2d image to a 1d list

if error, returns -1

origAddress + (width * row + col)
*/
__device__ int getPixelVal(int* image, int height, int width, int row, int col) {
    if (col < height && row < width && col >= 0 && row >= 0)
        return *(image + width * row + col);
    else{
        printf("CRAP");
        return -1;
    }
}

// helper functions -------------------------------------------------------------------

int* AllocateDeviceMemory (int* matrix, int size){
    int* res;

    cudaMalloc(&res, size);
    return res;
}

void CopyFromDevice(void* mDevice, void* mHost, int size){
    cudaMemcpy(mHost, mDevice, size, cudaMemcpyDeviceToHost);
}

void CopyToDevice(void* mHost,  void* mDevice, int size){
    cudaMemcpy(mDevice, mHost, size, cudaMemcpyHostToDevice);
}

