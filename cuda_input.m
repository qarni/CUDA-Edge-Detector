filenames = ["pngtest16rgba", "pnggrad16rgb", "pattern-large2", "bullseye", "gates", "gates-square"];
image_filename = strcat("~/Documents/School/468/Canny/test_images/", filenames(6), ".png");

im = imread(image_filename);
if size(im, 3) == 3
    bw = rgb2gray(im);
else
    bw = im;
end
% imshow(bw);

text_filename = strcat("~/Documents/School/468/Canny/matlab_matrix_images/", filenames(6), ".txt");

sizeim = size(bw);

dlmwrite(text_filename, sizeim,'delimiter','\t');
dlmwrite(text_filename, bw ,'delimiter','\t','newline','pc', '-append');