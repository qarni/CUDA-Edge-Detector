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
int** readImageMatrix(const char * filename, int * width, int * height);
int* readImageMatrix1D(const char * filename, int * width, int * height);

int main(int argc, const char **argv)
{
    if (argc != 2) {
        cout << "need to include name of image file" << endl;
        return -1;
    }

    cout << "Proceeding edge detection with file: " << argv[1] << endl;

    int width, height;

    // get image as a 2D int array
    //int** input = readImageMatrix(argv[1], &width, &height);
    //int** output = allocateMatrix(height, width);

    int* input = readImageMatrix1D(argv[1], &width, &height);
    int* output = allocateMatrix(height, width);

     for (int i = 0; i < height; i++)
    {
       for (int j = 0; j < width; j++)
       {
           output[width * i + j]= -1;
       }
       
    }

    cout << endl << endl;

    canny(input, height, width, output, 5, 1);

    // test
    cout << "in " << input[1] << " out " << output[1] << endl;

    int32_t count = 0;
    //test 2
    // for (int i = 0; i < height; i++)
    // {
    //    for (int j = 0; j < width; j++)
    //    {

    //        cout << output[width * i + j] << "-" << input[width * i + j] << " ";
    //        if(output[width * i + j] == -2) {
    //         count++;
    //        }
    //    }
       
    // }

    cout << endl << "count: " << count << endl;

    //test 3
    for (int i = 0; i < height; i++)
    {
       for (int j = 0; j < width; j++)
       {
           if(input[width * i + j] != output[width * i + j]) {
               cout << "FAILED" << " i " << i << " j " << j << endl;
               goto jump;
           }
       }
    }
    cout << "PASSED DUDE" << endl;

    jump:
    

    // free matrices
    // for (int i=0; i < width; i++){
    //     free(input[i]);
    //     free(output[i]);
    // }
    free(input);
    free(output);

    return 0;
}


int* allocateMatrix(int height, int width) {
    // int** matrix = (int **) malloc(sizeof(int *) * (width)); 
    // for (int i = 0; i < width; i++) 
        //  matrix[i] = (int*) malloc((height) * sizeof(int)); 

    int* matrix = (int *) malloc(sizeof(int) * width * height);
    return matrix;
}


/*
Turns image matrix text file into a 2d int array

first line of file is the width and height
following is pixel data

all is tab delimited with new lines for new rows

*/
/*
int** readImageMatrix(const char * filename, int * width, int * height) {
    string currline;
    ifstream file;
    string token;
    int t;

    file.open(filename);

    getline(file, currline, '\n'); 
    cout << currline << endl;

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

    int ** matrix = allocateMatrix(*height, *width);

    int row = 0;
    int col = 0;

    while(getline(file, currline, '\n')) {
        cout << "curr: " << currline << endl;
        istringstream ss3(currline);
        col = 0;
        int c = 0;
        while(getline(ss3, token, ' '))
        {
            if (c%2 ==0) {
            istringstream ss4(token);
            ss4 >> t;

            matrix[row][col] = t;
            cout << "mat: " << token << endl;
            col++;
            }
            c++;
        }
        row++;
    }

    // string currline;
    // ifstream file;
    // string token;
    // int t;
    
    // file.open(filename);

    // // first line, get the width and height
    // getline(file, currline, '\t'); 
    // stringstream linestream(currline);
    // getline(linestream, token,'\t');
    // stringstream ss(token);
    // ss >> t;
    // *width = t;
    // getline(linestream, token,'\t');
    // stringstream ss2(token);
    // ss2 >> t;
    // *height = t;

    // cout << "width: " << *width << " height: " << *height << endl;

    // int ** matrix = allocateMatrix(*height, *width);

    // int row = 0;
    // int col = 0;

    // cout << "ok!" << endl;

    // getline(file, currline);    // throw away for some reason 
    
    // cout << currline << endl;

    // // while(getline(file, currline, '\n')) {
    //     stringstream ss4(currline);
    //     col= 0;
    //     cout << currline << endl;
    //     while(getline(ss4, token,'\t'))
    //     {
    //         stringstream ss(token);
    //         ss >> t;
    //         matrix[row][col] = t;
    //         cout << "mat: " << matrix[row][col] << " ";
    //         col++;
    //     }
    //     row++;

    // // }

    // cout << "done" << row << col << endl;

    file.close();
    return matrix;
}
*/
int* readImageMatrix1D(const char * filename, int * width, int * height) {
    // string currline;
    // ifstream file;
    // string token;
    // int t;

    // file.open(filename);

    // getline(file, currline, '\n'); 
    // cout << currline << endl;

    // stringstream linestream(currline);
    // getline(linestream, token,'\t');
    // stringstream ss(token);
    // ss >> t;
    // *width = t;
    // getline(linestream, token,'\t');
    // stringstream ss2(token);
    // ss2 >> t;
    // *height = t;

    // cout << "width: " << *width << " height: " << *height << endl;

    // int * matrix = allocateMatrix(*height, *width);

    // int row = 0;
    // int col = 0;

    // while(getline(file, currline, '\n')) {
    //     cout << "curr: " << currline << endl;
    //     istringstream ss3(currline);
    //     col = 0;
    //     int c = 0;
    //     while(getline(ss3, token, ' '))
    //     {
    //         if (c%2 ==0) {
    //         istringstream ss4(token);
    //         ss4 >> t;

    //         matrix[*width * row + col] = t;
    //         cout << "mat: " << token << endl;
    //         col++;
    //         }
    //         c++;
    //     }
    //     row++;
    // }

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

    int * matrix = allocateMatrix(*height, *width);

    int row = 0;
    int col = 0;

    cout << "ok!" << endl;

    getline(file, currline);    // throw away for some reason 
    
    // cout << currline << endl;

    while(getline(file, currline, '\n')) {
        stringstream ss4(currline);
        col= 0;
        while(getline(ss4, token,'\t'))
        {
            stringstream ss(token);
            ss >> t;
            matrix[*width * row + col] = t;
            // cout << "mat: " << matrix[*width * row + col] << " ";
            col++;
        }
        row++;

    }

    cout << "done" << row << col << endl;

    file.close();
    return matrix;
}