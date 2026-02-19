f1 = imread("6835_.jpg");
f2 = imread("tile_size_2.jpg");
f3 = imread("tile_size_4.jpg");
f4 = imread("tile_size_8.jpg");
f5 = imread("tile_size_16.jpg");
f6 = imread("tile_size_32.jpg");

f1=reshape([f1,f1,f1], size(f1,1), size(f1,2), 3);

rx = 619+[0:331]; lx=diff(rx([1,end]))+1;
ry = 519+[0:331]; ly=diff(ry([1,end]))+1;
vbar = ones(ly,3,3)*255;
hbar = ones(3,lx*3+2*3,3)*255;
imshow([f1(ry,rx,:),vbar,f2(ry,rx,:),vbar,f3(ry,rx,:); hbar; ...
      f4(ry,rx,:),vbar,f5(ry,rx,:),vbar,f6(ry,rx,:)]);
print("singleScaleDetail", "-dpng");