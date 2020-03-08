#ifndef _CANNY_H_
#define _CANNY_H_

int canny(int* input, int* gaussianBlur, int* Ix, int* Iy, int* gradientMag, int* output, int height, int width, int kernelSize,  int sigma);
float*  generateGaussianFilter(int kernelSize, int sigma);

void* AllocateDeviceMemory (int size);
void CopyFromDevice(void* mDevice, void* mHost, int size);
void CopyToDevice(void* mHost,  void* mDevice, int size);

#endif
