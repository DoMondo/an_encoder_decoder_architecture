N=128; M = 8;
base = ones(N);
col = 0; i = 1;
while i <=N
    j = i+M/2+randi(M);
    base(:, i:j-1) = col;
    col = rem(col+1, 2);
    i = j;
end

% 30 = 180*0.1667
guess = imrotate(base, 2/6*180, 'bilinear', 'loose');

clf;
pos0 = [1:8]+69; pos1 = pos0; for i=1:7; pos1 = [pos0; pos1+69]; end;
angle=-1/12:-1/12:-1/2-1/12;
for i=1:7
    rotated = imrotate(guess, angle(i)*180, 'bilinear', 'loose');
    rotated = imresize(rotated, 0.9, 'bilinear');
    N2 = round(sum(size(rotated))/4);
    rotated = rotated(N2+[-N/4+1:N/4], N2+[-N/4+1:N/4]);
    pos0 = pos1 + (i-1)*10;
    subplot(9,69, pos0(:));
    imshow(rotated); axis('tight'); axis('equal'); 
    subplot(9,69,pos0(1)-69+[0:7]); plot(sum(rotated),'b', 'LineWidth', 2); ylim([0,N/2]); axis off;
    subplot(9,69,pos0(:,8)'+1); plot(sum(rotated'),'r', 'LineWidth', 2); camroll(-90); axis('tight'); axis off; ylim(1/3*[N/2,2*N/2]); % set(gca,'YDir','reverse'); 
    
end
% 
% figure;
% I = imshow(guess); hold on; B = plot(sum(rotated)); rotate(B, [0 1 0], 15);