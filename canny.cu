#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cutil.h>
#include "canny.h"

// kernel function definitions

// device function defintions 
__device__ int getPixelVal(int* image, int height, int width, int x, int y);
__global__ void GaussianBlur(int* input, int* output, int height, int width, float* gaussianFilter, int kernelSize, int32_t* count);
__global__ void FindGradients(int* input, int* output, int height, int width);

// ------------------------------------------------------------------------------------

/*
Wrapper function to make kernel calls to perform canny algorithm 
*/
void canny(int* input, int height, int width, int* output, int kernelSize,  int sigma) {

    clock_t before = clock();

    int matrixSize = height * width * sizeof(int);
    float* filter  = generateGaussianFilter(kernelSize, sigma);

    // set up for kernel calls 
    float* filterD;
    int32_t count = 0;
    int32_t* countD;

    int* inputD = AllocateDeviceMemory(matrixSize);
    int* gaussianBlurD = AllocateDeviceMemory(matrixSize);
    int* outputD = AllocateDeviceMemory(matrixSize);
    cudaMalloc(&filterD, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&countD, sizeof(int32_t));
    
    CopyToDevice(&(input[0]), inputD, matrixSize);
    CopyToDevice(&(output[0]), outputD, matrixSize);
    CopyToDevice(&(filter[0]), filterD, kernelSize * kernelSize * sizeof(float));
    CopyToDevice(&count, countD, sizeof(int32_t));

    // set up dimensions for calls to kernel -------------------------------------------------------------------

    // 2400 = 8  * 300
    // 600  = 8 * 75
    // 4 = 1  * 4

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(300, 300);

    GaussianBlur<<<numBlocks,threadsPerBlock>>>(inputD, gaussianBlurD, height, width, filterD, kernelSize, countD);
    cudaThreadSynchronize();

    FindGradients<<<numBlocks, threadsPerBlock>>>(gaussianBlurD, outputD, height, width);
    cudaThreadSynchronize();

    // tear down after kernel calls are done -------------------------------------------------------------------
    CopyFromDevice(outputD, &(output[0]), matrixSize);
    CopyFromDevice(countD, &count, sizeof(int32_t));
    cudaFree(inputD);
    cudaFree(outputD);
    cudaFree(filterD);
    cudaFree(countD);

    printf("\n\ndone with canny algorithm\n");
    printf("count: %d\n", count);

    clock_t difference = clock() - before;
    int msec = difference * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

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

For the borders of the image (anything less than the size of the kernel away from an edge)
Just use the original vals of the image

*/
__global__ void GaussianBlur(int* input, int* output, int height, int width, float* gaussianFilter, int kernelSize, int32_t* count) {

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    output[width * row + col] = -5;

    int val = getPixelVal(input, height, width, row, col);
    if(val == -1){
        return;
    }
        
    int kernelHalf = kernelSize/2;

    // account for borders of the image which can't have the filter applied to them
    if(row < kernelHalf || col < kernelHalf || row > width - 1 - kernelHalf || col > height - 1 - kernelHalf) {
        output[width * row + col] = val;
    }
    // otherwise, apply the filter!
    else {

        float filteredVal = 0.0;
        int f = 0;
        for(int krow = -kernelHalf; krow <= kernelHalf; krow++) {
            for(int kcol = -kernelHalf; kcol <= kernelHalf; kcol++) {
                filteredVal += (float)getPixelVal(input, height, width, row + krow, col + kcol) * gaussianFilter[f];
                f++;
            }
        }
        
        output[width * row + col] = (int)filteredVal;
    }

    __syncthreads();

    atomicAdd(count, 1);
}

/*
Find gradients - this is the step that actually detects edges (roughly)

Very similar to previous step, just need to apply Sobel filters this time

Kx = -1 0 1 -2 0 2 -1 0 1
Ky = 1 2 1 0 0 0 -1 -2 -1

Also need this data for later:

Magnitude G = sqrt(Ix^2 + Iy^2)
slope O grad = arctan(Iy/Ix)

*/
__global__ void FindGradients(int* input, int* output, int height, int width) {

    // sobel filters. Apply both!
    int Kx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int Ky[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    int val = getPixelVal(input, height, width, row, col);
    if(val == -1){
        return;
    }

    // account for borders of the image which can't have the filter applied to them
    if(row < 1 || col < 1 || row > width - 2 || col > height - 2) {
        output[width * row + col] = val;
    }
    // otherwise, apply the filters!
    else {

        float filteredValX = 0.0;
        float filteredValY = 0.0;
        int f = 0;
        for(int krow = -1; krow <= 1; krow++) {
            for(int kcol = -1; kcol <= 1; kcol++) {
                filteredValX += (float)getPixelVal(input, height, width, row + krow, col + kcol) * Kx[f];
                filteredValY += (float)getPixelVal(input, height, width, row + krow, col + kcol) * Ky[f];
                f++;
            }
        }

        float sobel = sqrt(pow(filteredValX, 2) + pow(filteredValY, 2));
        
        output[width * row + col] = (int)sobel;

        // TODO: calc gradient direction
    }

    __syncthreads();

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

int* AllocateDeviceMemory (int size){
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

