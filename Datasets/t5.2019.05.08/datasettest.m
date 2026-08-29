data = load('singleLetters.mat');
size(data.neuralActivityCube_a)
cube = data.neuralActivityCube_a;   % K x T x N
avgActivity = squeeze(mean(cube, 1)); % T x N, average across trials

figure;
imagesc(avgActivity');  % electrodes vs time
xlabel('Time (ms)');
ylabel('Electrode');
colorbar;
title('Average spike activity for character A');