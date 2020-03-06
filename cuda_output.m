M=dlmread("~/Desktop/output.txt")

M = uint8(M)

figure(1);
imshow(bw)
figure(2);
imshow(M)