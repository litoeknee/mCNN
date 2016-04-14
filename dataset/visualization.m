%% Initialization
clear ; close all; clc
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data

fprintf('Loading and Visualizing Data ...\n')

X = load('mnist_train_1000.csv');
m = size(X, 1);

% Randomly select 100 data points to display
ransel = randperm(size(X, 1));
for i = 0:1:m/100
    sel = ransel(1 + i*100:(i + 1)*100);
    displayData(X(sel, 2:size(X, 2)));
    
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end