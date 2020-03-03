#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstdlib>

#include <cuda_runtime.h>

using namespace std;

int** readImageMatrix(const char * filename, int * width, int * height);

int main(int argc, const char **argv)
{
    if (argc != 2)
        cout << "need to include name of image file" << endl;

    cout << "Proceeding edge detection with file: " << argv[1] << endl;

    int** matrix;
    int width, height;

    matrix = readImageMatrix(argv[1], &width, &height);

    for (int i=0; i < width; i++)
        free(matrix[i]);
    free(matrix);

    return 0;
}


/*
Turns image matrix text file into a 2d int array

first line of file is the width and height
following is pixel data

all is tab delimited with new lines for new rows

*/
int** readImageMatrix(const char * filename, int * width, int * height) {

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

    cout << "width: " << *width << " height: " << *height << endl;

    int** matrix = (int **) malloc(sizeof(int *) * (*width)); 
    for (int i = 0; i < *width; i++) 
         matrix[i] = (int*) malloc((*height) * sizeof(int)); 

    int row = 0;
    int col = 0;

    getline(file, currline);    // throw away for some reason 

    while(getline(file, currline)) {
        stringstream linestream(currline);
        col= 0;
        while(getline(linestream, token,'\t'))
        {
            stringstream ss(token);
            ss >> t;
            matrix[row][col] = t;
            col++;
        }
        row++;

    }

    file.close();
    return matrix;
}