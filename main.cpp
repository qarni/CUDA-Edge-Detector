#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstdlib>

#include <cuda.h>
#include <cutil.h>

#include "canny.h"

using namespace std;

int* allocateMatrix(int height, int width);
int* readImageMatrix1D(const char * filename, int * width, int * height);
void writeMatrixToFile(int* matrix, const char * filename, int width, int height);

int main(int argc, const char **argv)
{
    int KERNEL_SIZE = 5;
    int SIGMA = 1;
    double LOWER_THRESHOLD = 0.05;
    double UPPER_THRESHOLD = 0.09;

    if (argc != 2) {
        if (argc < 2) {
            cout << "need to include name of image file" << endl;
            return -1;
        }
        else if (argc == 6) {
            stringstream ss(argv[2]);
            ss >> KERNEL_SIZE;
            stringstream ss2(argv[3]);
            ss2 >> SIGMA;
            stringstream ss3(argv[4]);
            ss3 >> LOWER_THRESHOLD;
            stringstream ss4(argv[5]);
            ss4 >> UPPER_THRESHOLD;
        }
        else {
            cout << "Parameters are wrong" << endl;
            return -1;
        }
    }

    cout << endl << "Proceeding edge detection with file: " << argv[1] << endl;
    cout << "Kernel size: " << KERNEL_SIZE << " Sigma: " << SIGMA << " Lower threshold: " << LOWER_THRESHOLD << " Upper threshold: " << UPPER_THRESHOLD << endl;

    int width, height;

    // get image as a 1D int array
    int* input = readImageMatrix1D(argv[1], &width, &height);

    int* gaussianBlur = allocateMatrix(height, width);
    int* Ix = allocateMatrix(height, width);
    int* Iy = allocateMatrix(height, width);
    int* gradientMag = allocateMatrix(height, width);
    int* nonMaximumSupression = allocateMatrix(height, width);
    int* doubleThreshold = allocateMatrix(height, width);
    int* output = allocateMatrix(height, width);

    for (int i = 0; i < height; i++)
    {
       for (int j = 0; j < width; j++)
       {
           output[width * i + j]= -1;
       }
    }

    // do the thing!
    if(canny(input, gaussianBlur, Ix, Iy, gradientMag, nonMaximumSupression, doubleThreshold, output, height, width, KERNEL_SIZE, SIGMA, LOWER_THRESHOLD, UPPER_THRESHOLD) != -1) {
        // write output to a file so we can open as an image in matlab!
        // write each step so we can compare
        writeMatrixToFile(input, "output_matrix/input.txt", width, height);
        writeMatrixToFile(gaussianBlur, "output_matrix/gaussianBlur.txt", width, height);
        writeMatrixToFile(Ix, "output_matrix/Ix.txt", width, height);
        writeMatrixToFile(Iy, "output_matrix/Iy.txt", width, height);
        writeMatrixToFile(gradientMag, "output_matrix/gradientMag.txt", width, height);
        writeMatrixToFile(nonMaximumSupression, "output_matrix/nonMaximumSupression.txt", width, height);
        writeMatrixToFile(doubleThreshold, "output_matrix/strongEdges.txt", width, height);
        writeMatrixToFile(output, "output_matrix/output.txt", width, height);

        // test to make sure output doesnt have a -1
        cout << "in " << input[1] << " out " << output[1] << endl;
    }
    else {
        cout << "something went wrong :(" << endl;
    }
    
    free(input);
    free(gaussianBlur);
    free(Ix);
    free(Iy);
    free(gradientMag);
    free(nonMaximumSupression);
    free(doubleThreshold);
    free(output);

    cout << "goodbye" << endl;
    return 0;
}


int* allocateMatrix(int height, int width) {
    int* matrix = (int *) malloc(sizeof(int) * width * height);
    return matrix;
}


/*
Turns image matrix text file into a 1d int array

first line of file is the width and height
following is pixel data

all is tab delimited with new lines for new rows

*/
int* readImageMatrix1D(const char * filename, int * width, int * height) {

    string currline;
    ifstream file;
    string token;
    int t;
    
    file.open(filename);

    // first line, get the width and height
    getline(file, currline, '\t'); 
    stringstream linestream(currline);
    getline(linestream, token,'\t');
    stringstream ss(token);
    ss >> t;
    *width = t;
    getline(linestream, token,'\t');
    stringstream ss2(token);
    ss2 >> t;
    *height = t;

    // TODO: not reading height correctly
    cout << "width: " << *width << " height: " << *height << endl;

    int * matrix = allocateMatrix(*height, *width);

    int row = 0;
    int col = 0;

    getline(file, currline);    // throw away for some reason 
    
    while(getline(file, currline, '\n')) {
        stringstream ss4(currline);
        col= 0;
        while(getline(ss4, token,'\t'))
        {
            stringstream ss(token);
            ss >> t;
            matrix[*width * row + col] = t;
            col++;
        }
        row++;

    }

    file.close();
    return matrix;
}

/*
Turns 1d int array (representing a 2d image) into a file

tab delimited with new lines for new rows
*/
void writeMatrixToFile(int* matrix, const char * filename, int width, int height) {
   
    ofstream file (filename);
        for (int row = 0; row < width; row++) {
            for(int col = 0; col < height; col++) {
                if(col == height - 1)
                    file << matrix[width * row + col] << "\n";
                else
                    file << matrix[width * row + col] << "\t";
            }
        }
        file.close();

}