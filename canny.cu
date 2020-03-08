#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cutil.h>
#include "canny.h"

// kernel function definitions
__global__ void GaussianBlur(int* input, int* output, float* gaussianFilter, int kernelSize, int32_t* count, int height, int width);
__global__ void FindGradients(int* input, int* output, int* gradientDir, int* Ix, int* Iy, int height, int width);
__global__ void NonMaximumSuppression(int* input, int* output, int* gradientDir,  int* Ix, int* Iy, int height, int width);

// device function defintions 
__device__ int getPixelVal(int* image, int height, int width, int x, int y);

// ------------------------------------------------------------------------------------

/*
* Wrapper function to make kernel calls to perform canny algorithm 
*/
int canny(int* input, int* gaussianBlur, int* Ix, int* Iy, int* gradientMag, int* nonMaximumSuppressed, int* doubleThreshold, int* output, 
    int height, int width, int kernelSize,  int sigma, double lowerThreshold, double upperThreshold) {

    clock_t before = clock();

    int matrixSize = height * width * sizeof(int);
    float* filter  = generateGaussianFilter(kernelSize, sigma);

    int32_t count = 0;

    int* inputD = (int*)AllocateDeviceMemory(matrixSize);               // print
    
    int* gaussianBlurD = (int*)AllocateDeviceMemory(matrixSize);        // print
    int* gradientDirD = (int*)AllocateDeviceMemory(matrixSize);
    int* IxD = (int*)AllocateDeviceMemory(matrixSize);                  // print
    int* IyD = (int*)AllocateDeviceMemory(matrixSize);                  // print

    int* gradientMagnitudeD = (int*)AllocateDeviceMemory(matrixSize);   // print

    int* nonMaxSuppressedD = (int*)AllocateDeviceMemory(matrixSize);    // print

    int* doubleThresholdD = (int*)AllocateDeviceMemory(matrixSize);     // print

    int* outputD = (int*)AllocateDeviceMemory(matrixSize);              // print
    
    float* filterD = (float*)AllocateDeviceMemory(kernelSize * kernelSize * sizeof(float));
    int32_t* countD = (int32_t*)AllocateDeviceMemory(sizeof(int32_t));
    
    CopyToDevice(&(input[0]), inputD, matrixSize);
    CopyToDevice(&(gaussianBlur[0]), gaussianBlurD, matrixSize);
    CopyToDevice(&(Ix[0]), IxD, matrixSize);
    CopyToDevice(&(Iy[0]), IyD, matrixSize);
    CopyToDevice(&(gradientMag[0]), gradientMagnitudeD, matrixSize);
    CopyToDevice(&(nonMaximumSuppressed[0]), nonMaxSuppressedD, matrixSize);
    CopyToDevice(&(doubleThreshold[0]), doubleThresholdD, matrixSize);
    CopyToDevice(&(output[0]), outputD, matrixSize);

    CopyToDevice(&(filter[0]), filterD, kernelSize * kernelSize * sizeof(float));
    CopyToDevice(&count, countD, sizeof(int32_t));

    // set up dimensions for calls to kernel -------------------------------------------------------------------

    // 2400 = 8  * 300
    // 600  = 8 * 75
    // 4 = 1  * 4
    // 600x384 = 8,8 x 75, 48
    // 384 = 8 * 48

    // TODO: cleanup later
    // only works for images that are square and divisible by 64
    if (height != width) {
        printf("not supported rn\n");
        return -1;
    }

    dim3 threadsPerBlock(1,1);

    if ((height * width) % 64 == 0) 
        dim3 threadsPerBlock(8, 8);
    else if ((height * width) % 16 == 0)
        dim3 threadsPerBlock(4, 4); 
    else if ((height * width) % 4 == 0)
        dim3 threadsPerBlock(2, 2);
    else if ((height * width) % 1 == 0)
        dim3 threadsPerBlock(1, 1);
    else {
        printf("literally how? \n");
        return -1;
    }

    dim3 numBlocks(height/threadsPerBlock.x, width/threadsPerBlock.y);

    // make kernel calls ---------------------------------------------------------------------------------------

    GaussianBlur<<<numBlocks,threadsPerBlock>>>(inputD, gaussianBlurD, filterD, kernelSize, countD, height, width);
    cudaThreadSynchronize();

    FindGradients<<<numBlocks, threadsPerBlock>>>(gaussianBlurD, gradientMagnitudeD, gradientDirD, IxD, IyD, height, width);
    cudaThreadSynchronize();

    NonMaximumSuppression<<<numBlocks, threadsPerBlock>>>(gradientMagnitudeD, outputD, gradientDirD, IxD, IyD, height, width);
    cudaThreadSynchronize();

    // tear down after kernel calls are done -------------------------------------------------------------------
    CopyFromDevice(gaussianBlurD, &(gaussianBlur[0]), matrixSize);
    CopyFromDevice(IxD, &(Ix[0]), matrixSize);
    CopyFromDevice(IyD, &(Iy[0]), matrixSize);
    CopyFromDevice(gradientMagnitudeD, &(gradientMag[0]), matrixSize);
    CopyFromDevice(nonMaxSuppressedD, &(nonMaximumSuppressed[0]), matrixSize);
    CopyFromDevice(doubleThresholdD, &(doubleThreshold[0]), matrixSize);
    CopyFromDevice(outputD, &(output[0]), matrixSize);
    CopyFromDevice(countD, &count, sizeof(int32_t));

    cudaFree(inputD);
    cudaFree(gaussianBlurD);
    cudaFree(IxD);
    cudaFree(IyD);
    cudaFree(gradientMagnitudeD);
    cudaFree(nonMaxSuppressedD);
    cudaFree(doubleThresholdD);
    cudaFree(outputD);
    cudaFree(filterD);
    cudaFree(countD);

    printf("\ndone with canny algorithm\n\n");

    clock_t difference = clock() - before;
    int msec = difference * 1000 / CLOCKS_PER_SEC;
    printf("Time taken: approx %d second(s) (%d milliseconds)\n", msec/1000, msec%1000);
    if(msec < 1000) 
        printf("oh that is fast!\n");

    return 0;
}


// kernel functions --------------------------------------------------------------------

/*
* Noise reduction - gets rid of background noise but still keeps borders more in focus so they 
* can be detected in the next step
*
* Apply gaussian filter over each pixel 
* Start with kernel size of 5?
*
* For the borders of the image (anything less than the size of the kernel away from an edge)
* Just use the original vals of the image
*
*/
__global__ void GaussianBlur(int* input, int* output, float* gaussianFilter, int kernelSize, int32_t* count, int height, int width) {

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
* Find gradients - this is the step that actually detects edges (roughly)
*
*Very similar to previous step, just need to apply Sobel filters this time
*
* Kx = -1 0 1 -2 0 2 -1 0 1
* Ky = 1 2 1 0 0 0 -1 -2 -1
*
* Magnitude G = sqrt(Ix^2 + Iy^2)
*
* Also need this data for later:
* slope O grad = arctan(Iy/Ix)
*/
__global__ void FindGradients(int* input, int* output, int* gradientDir, int* Ix, int* Iy, int height, int width) {

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

        Ix[width * row + col] = filteredValX;
        Iy[width * row + col] = filteredValY;

        float sobel = sqrt(pow(filteredValX, 2) + pow(filteredValY, 2));
        
        // sobel filter output
        output[width * row + col] = (int)sobel;

        // calc gradient direction
        // convert from radians to degrees
        gradientDir[width * row + col] = (int)(atan(filteredValX/filteredValY) * 180 / M_PI);
    }

    __syncthreads();

}


/*
* The current image has thick and thin edges - the final image should have only thin edges
* Find pixels with the maximum value in edge directions
* 
* If pixel has a higher magnitude than either of the pixels in its direction, keep pixel
* Otherwise just make it black (0)
* Uses interpolation with Ix from previous step
*/
__global__ void NonMaximumSuppression(int* input, int* output, int* gradientDir, int* Ix, int* Iy, int height, int width) {

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    int gradientAngle = getPixelVal(gradientDir, height, width, row, col);
    if(gradientAngle == -1){
        return;
    }

    int gradientMag = getPixelVal(input, height, width, row, col);

    // account for borders of the image which can't have calculations done 
    if(row < 1 || col < 1 || row > width - 2 || col > height - 2) {
        output[width * row + col] = gradientMag;
    }
    // otherwise, do it
    else {
        int up1, up2, down1, down2;
        int est;

        // make sure that the angle is greater than 0 (not a negative angle) to make life easier for calculations
        if (gradientAngle < 0) 
            gradientAngle += 180;

        // angle 0 - 45
        if(gradientAngle >= 0 && gradientAngle < 45) { 

            up1 = getPixelVal(input, height, width, row, col - 1);
            up2 = getPixelVal(input, height, width, row - 1, col - 1);
            down1 = getPixelVal(input, height, width, row, col + 1);
            down2 = getPixelVal(input, height, width, row + 1, col + 1);

            est = abs(getPixelVal(Ix, height, width, row, col) / gradientMag);
        }
        // angle 45 - 90
        else if(gradientAngle >= 45 && gradientAngle < 90) {

            up1 = getPixelVal(input, height, width, row - 1, col);
            up2 = getPixelVal(input, height, width, row - 1, col - 1);
            down1 = getPixelVal(input, height, width, row, col + 1);
            down2 = getPixelVal(input, height, width, row + 1, col + 1);

            est = abs(getPixelVal(Ix, height, width, row, col) / gradientMag);
        }
        // angle 90 - 135
        else if(gradientAngle >= 90 && gradientAngle < 135) {

            up1 = getPixelVal(input, height, width, row - 1, col);
            up2 = getPixelVal(input, height, width, row - 1, col + 1);
            down1 = getPixelVal(input, height, width, row + 1, col);
            down2 = getPixelVal(input, height, width, row + 1, col - 1);

            est = abs(getPixelVal(Ix, height, width, row, col) / gradientMag);
        }
        // angle 135 - 180
        else if(gradientAngle >= 135 && gradientAngle < 180) {

            up1 = getPixelVal(input, height, width, row, col + 1);
            up2 = getPixelVal(input, height, width, row - 1, col + 1);
            down1 = getPixelVal(input, height, width, row, col - 1);
            down2 = getPixelVal(input, height, width, row + 1, col - 1);

            est = abs(getPixelVal(Ix, height, width, row, col) / gradientMag);
        }

        if ((gradientMag >= (down2-down1) * est + down1) && (gradientMag >= (up2 - up1) * est + up1))
            output[width * row + col] = gradientMag;
        else
            output[width * row + col] = 0;
    
    }
}

// device functions --------------------------------------------------------------------
// can only be called from global func or from another device func, not from host

/*
* returns pixel value at a location 
* Maps a 2d image to a 1d list
*
* if error, returns -1
*
* origAddress + (width * row + col)
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

/*
* This is the guassian filter to be applied over each pixel in the image
*
* G(x, y) = (1/2*pi*sigma^2)*e^-(x^2+y^2/2*sigma^2)
*/
float* generateGaussianFilter(int kernelSize, int sigma) {

    float* filter  = (float*) malloc(kernelSize * kernelSize * sizeof(float));
    int kernelHalf = kernelSize/2;

    float div = 2.0 * sigma *  sigma;
    float pre = 1.0 / (M_PI * div);

    int i  = 0;
    
    for (int x = -kernelHalf; x <= kernelHalf; x++) { 
        for (int y = -kernelHalf; y <= kernelHalf; y++) { 
            filter[i] = pre * pow(M_E, -((pow(x,2) + pow(y, 2)) / div));
            i++;
        }
    }

    return filter;
}

void* AllocateDeviceMemory (int size){
    void* res;
    cudaMalloc(&res, size);
    return res;
}

void CopyFromDevice(void* mDevice, void* mHost, int size){
    cudaMemcpy(mHost, mDevice, size, cudaMemcpyDeviceToHost);
}

void CopyToDevice(void* mHost,  void* mDevice, int size){
    cudaMemcpy(mDevice, mHost, size, cudaMemcpyHostToDevice);
}

