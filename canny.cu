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
__global__ void DoubleThreshold(int* input, int* output, int max, double lowerThresholdRatio, double upperThresholdRatio, int height, int width);
__global__ void Hysteresis(int* nonMaximumSuppressed, int* intensity, int* output, int height, int width);

// device function defintions 
__device__ int getPixelVal(int* image, int height, int width, int x, int y);

// ------------------------------------------------------------------------------------

/*
* Wrapper function to make kernel calls to perform canny algorithm 
*/
int canny(int* input, int* gaussianBlur, int* Ix, int* Iy, int* gradientMag, int* nonMaximumSuppressed, int* doubleThreshold, int* output, 
    int height, int width, int kernelSize, int sigma, double lowerThreshold, double upperThreshold) {

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

    dim3 threadsPerBlock(8,8);

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
    
    int space = (threadsPerBlock.x + kernelSize) * (threadsPerBlock.y + kernelSize);
    printf("space allocated: %d\n", space );

    GaussianBlur<<<numBlocks,threadsPerBlock, space * sizeof(int32_t)>>>(inputD, gaussianBlurD, filterD, kernelSize, countD, height, width);
    cudaThreadSynchronize();

    FindGradients<<<numBlocks, threadsPerBlock>>>(gaussianBlurD, gradientMagnitudeD, gradientDirD, IxD, IyD, height, width);
    cudaThreadSynchronize();

    NonMaximumSuppression<<<numBlocks, threadsPerBlock>>>(gradientMagnitudeD, nonMaxSuppressedD, gradientDirD, IxD, IyD, height, width);
    cudaThreadSynchronize();

    CopyFromDevice(nonMaxSuppressedD, &(nonMaximumSuppressed[0]), matrixSize);
    // get the maximum value
    int max = getMaxValue(nonMaximumSuppressed, height*width);

    DoubleThreshold<<<numBlocks, threadsPerBlock>>>(nonMaxSuppressedD, doubleThresholdD, max, lowerThreshold, upperThreshold, height, width);
    cudaThreadSynchronize();

    Hysteresis<<<numBlocks, threadsPerBlock>>>(nonMaxSuppressedD, doubleThresholdD, outputD, height, width);    
    cudaThreadSynchronize();

    // tear down after kernel calls are done -------------------------------------------------------------------
    CopyFromDevice(gaussianBlurD, &(gaussianBlur[0]), matrixSize);
    CopyFromDevice(IxD, &(Ix[0]), matrixSize);
    CopyFromDevice(IyD, &(Iy[0]), matrixSize);
    CopyFromDevice(gradientMagnitudeD, &(gradientMag[0]), matrixSize);
    
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

    // TODO: implement shared memory for input so we dont have to read it so many times (x6 per thread)
    // each thread in a block can load in one index of memory
    // the threads on the edges of the blocks can load in extra(?) or just read from input for those
    // blockId = block size / num threads, goes from 0 - this val
    // the index will be threadIdx.x + kernelSize, threadIdx.y + kernelSize

    // dynamically allocated by the kernel call to be of size: (numThreads.x + kernelSize) * (Numthreads.x + kernelSize)
    // which means each index will be (blockDim.x + kernelSize) * (threadIdx.x) + (threadIdx.y)
    extern __shared__ int32_t sharedMem[];

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    output[width * row + col] = -5;

    int val = getPixelVal(input, height, width, row, col);
    if(val == -1){
        return;
    }

    int kernelHalf = kernelSize/2;

    // load one pixel in per thread
    int s_width = blockDim.x + (kernelSize);
    int s_height = blockDim.y + (kernelSize);
    int s_row = threadIdx.x + kernelHalf;
    int s_col = threadIdx.y + kernelHalf;
    int currIndex = s_width * s_row + s_col;

    sharedMem[currIndex] = val;

    __syncthreads();

    // load extra for the borders...
    // add checks if the current block id is at the border of the full grid

    if (threadIdx.x == 0 && blockIdx.y != 0) {
        for (int i = kernelHalf; i > 0; i--){
            int cval = getPixelVal(input, height, width, row, col - i);
            if (cval != -1 )
                sharedMem[s_width * s_row + (s_col - i)] = cval;
            else
                sharedMem[s_width * s_row + (s_col - i)] = 0;
        }
    }
    if (threadIdx.y == 0 && blockIdx.x != 0) {
        for (int i = kernelHalf; i > 0; i--){
            int cval = getPixelVal(input, height, width, row - i, col);
            if (cval != -1)
                sharedMem[s_width * (s_row - i) + s_col] = cval;
            else
                sharedMem[s_width * (s_row - i) + s_col] = 0;
        }
    }
    if (threadIdx.x == blockDim.x - 1 && blockIdx.y != gridDim.y - 1) {
        for (int i = 0; i < kernelHalf; i++){
            int cval = getPixelVal(input, height, width, row, col + i);
            if (cval != -1)
                sharedMem[s_width * s_row + (s_col + i)] = cval;
            else
                sharedMem[s_width * s_row + (s_col + i)] = 0;
        }
    }
    if (threadIdx.y == blockDim.x - 1 && blockIdx.x != gridDim.x - 1) {
        for (int i = 0; i < kernelHalf; i++){
            int cval = getPixelVal(input, height, width, row + i, col);
            if (cval != -1)
                sharedMem[s_width * (s_row + i) + s_col] = cval;
            else
                sharedMem[s_width * (s_row + i) + s_col] = 0;
        }
    }

    __syncthreads();    // wait for all threads to finish


    // account for borders of the image which can't have the filter applied to them
    if(row < kernelHalf || col < kernelHalf || row > width - 1 - kernelHalf || col > height - 1 - kernelHalf) {
        output[width * row + col] = val;
    }
    else {
        
        // otherwise, apply the filter!
        float filteredVal = 0.0;
        float sharedVal = 0.0;
        int f = 0;
        for(int krow = -kernelHalf; krow <= kernelHalf; krow++) {
            for(int kcol = -kernelHalf; kcol <= kernelHalf; kcol++) {
                filteredVal += (float) getPixelVal(input, height, width, row + krow, col + kcol) * gaussianFilter[f];
                sharedVal += (float) getPixelVal(sharedMem, s_height, s_width, s_row + krow, s_col + kcol) * gaussianFilter[f];

                if (getPixelVal(input, height, width, row + krow, col + kcol) != getPixelVal(sharedMem, s_height, s_width, s_row + krow, s_col + kcol)) {
                    printf("%d-%d r %d %d c %d %d\n", getPixelVal(input, height, width, row + krow, col + kcol), getPixelVal(sharedMem, s_height, s_width, s_row + krow, s_col + kcol), row, s_row, col, s_col);
                }
                f++;
            }
        }

        // if(sharedVal != filteredVal)
        //     printf("DOES NOT MATCH %lf %lf |", sharedVal, filteredVal);

        
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
        output[width * row + col] = 0;
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

    __syncthreads();
}

/*
* Mark each pixel as strong, weak or irrelevant
* irrelevant pixels are turned black if they arent already
* Weak pixels are examined in the next step
* 
* strong = 255
* weak = 1
* irrelevant = 0
*/
__global__ void DoubleThreshold(int* input, int* output, int max, double lowerThresholdRatio, double upperThresholdRatio, int height, int width) {

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    int val = getPixelVal(input, height, width, row, col);
    if(val == -1){
        return;
    }

    int upperThresholdVal = max * upperThresholdRatio;
    int lowerThresholdVal = upperThresholdVal * lowerThresholdRatio;

    if (val > upperThresholdVal)
        output[width * row + col] = 255;    // strong
    else if (val > lowerThresholdVal)
        output[width * row + col] = 1;      // weak
    else
        output[width * row + col] = 0;      // irrelevant

    __syncthreads();
}

/*
* Get rid of irrelevant pixels (make 0)
* Keep strong pixels
* Weak pixel: only keep if one of the surrounding pixels is strong
*/
__global__ void Hysteresis(int* nonMaximumSuppressed, int* intensity, int* output, int height, int width) {

    int STRONG = 255;
    int WEAK = 1;
    int IRRELEVANT = 0;

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    int val = getPixelVal(nonMaximumSuppressed, height, width, row, col);
    int strength = getPixelVal(intensity, height, width, row, col);
    if(val == -1){
        return;
    }

    // account for borders of the image which can't have calculations done 
    if(row < 1 || col < 1 || row > width - 2 || col > height - 2) {
        output[width * row + col] = 0;
    }
    // otherwise, do it
    else {
        if (strength == STRONG) {
            output[width * row + col] = val;
        }
        else if (strength == IRRELEVANT)
            output[width * row + col] = 0;
        else if (strength == WEAK) {
            int found = 0;
            if (getPixelVal(intensity, height, width, row - 1, col - 1) == STRONG)
                found = 1;
            else if (getPixelVal(intensity, height, width, row - 1, col) == STRONG)
                found = 1;
            else if (getPixelVal(intensity, height, width, row - 1, col + 1) == STRONG)
                found = 1;
            else if (getPixelVal(intensity, height, width, row, col - 1) == STRONG)
                found = 1;
            else if (getPixelVal(intensity, height, width, row - 1, col + 1) == STRONG)
                found = 1;
            else if (getPixelVal(intensity, height, width, row + 1, col - 1) == STRONG)
                found = 1;
            else if (getPixelVal(intensity, height, width, row + 1, col) == STRONG)
                found = 1;
            else if (getPixelVal(intensity, height, width, row + 1, col + 1)  == STRONG)
                found = 1;
            
            if (found == 1)
                output[width * row + col] = val;
            else
                output[width * row + col] = 0;
        }
    }

    __syncthreads();
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
        printf("CRAP height: %d width: %d row: %d col: %d\n", height, width, row, col);
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

int getMaxValue(int* input, int size) {
    int max = 0;
    
    for(int i  = 0; i < size; i++) {
        int curr = input[i];
        if (curr > max)
            max = curr;
    }

    return max;
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

