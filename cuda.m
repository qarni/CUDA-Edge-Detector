filenames = ["pngtest16rgba", "pnggrad16rgb"];
image_filename = strcat("~/Documents/School/468/CUDA Edge Detector/test_images/", filenames(2), ".png");

im = imread(image_filename);
bw = rgb2gray(im);
% imshow(bw);

text_filename = strcat("~/Documents/School/468/CUDA Edge Detector/matlab_matrix_images/", filenames(2), ".txt");

sizeim = size(bw);

dlmwrite(text_filename, sizeim,'delimiter','\t','newline', 'pc');
dlmwrite(text_filename, bw ,'delimiter','\t','newline','pc', '-append');