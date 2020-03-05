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
__global__ void GaussianBlur(int* input, int* output, int kernelSize);


// ------------------------------------------------------------------------------------

/*
Wrapper function to make kernel calls to perform canny algorithm 
*/
void canny(int** input, int height, int width, int** output, int kernelSize,  int sigma) {

    int matrixSize = height * width * sizeof(int);
    float* filter  = generateGaussianFilter(kernelSize, sigma);

    // set up for kernel calls 
    int* inputD = AllocateDeviceMemory(input, matrixSize);
    int* outputD = AllocateDeviceMemory(output, matrixSize);
    float* filterD;
    cudaMalloc(&filterD, kernelSize * kernelSize * sizeof(float));
    CopyToDevice(&(input[0][0]), inputD, matrixSize);
    CopyToDevice(&(input[0][0]), outputD, matrixSize);
    CopyToDevice(&filter, filterD, kernelSize * kernelSize * sizeof(float));





    // tear down after kernel calls are done
    CopyFromDevice(outputD, &(output[0][0]), matrixSize);
    cudaFree(inputD);
    cudaFree(outputD);
    cudaFree(filterD);

    printf("done with canny algorithm\n");
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
__global__ void GaussianBlur(int* input, int* output, float* gaussianFilter, int kernelSize) {



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
    else
        return -1;
}

// helper functions -------------------------------------------------------------------

int* AllocateDeviceMemory (int** matrix, int size){
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

