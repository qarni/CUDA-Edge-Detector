#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <png.h>

using namespace std;

void readInFile(string filename);

int main(int argc, const char *argv[])
{

    if (argc != 2)
    {
        cout << "need to include name of image file" << endl;
    }

    string filename = argv[1];
    cout << "Proceeding edge detection with file: " << filename << endl;

    readInFile(filename);

    return 0;
}

/*
Opens the png file

TODO:
1 - make sure that it is a png file
2 - make sure that it is black and white
3 - get the file dimensions 
*/
void readInFile(char* filename)
{

    int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

    // ifstream file;
    // file.open(filename);

    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    // if (!png)
    //     abort();

    // png_infop info = png_create_info_struct(png);
    // if (!info)
    //     abort();

    // if (setjmp(png_jmpbuf(png)))
    //     abort();

    // png_init_io(png, fp);

    // png_read_info(png, info);

    // width = png_get_image_width(png, info);
    // height = png_get_image_height(png, info);
    // color_type = png_get_color_type(png, info);
    // bit_depth = png_get_bit_depth(png, info);

}