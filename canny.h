#ifndef _CANNY_H_
#define _CANNY_H_

void canny(int* input, int height, int width, int* output, int kernelSize,  int sigma);
float*  generateGaussianFilter(int kernelSize, int sigma);

int* AllocateDeviceMemory (int size);
void CopyFromDevice(void* mDevice, void* mHost, int size);
void CopyToDevice(void* mHost,  void* mDevice, int size);

#endif
