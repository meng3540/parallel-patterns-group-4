% Load the CSV file
file_path = 'C:\Users\ferna\Desktop\Parallel Prog\histogram.csv'; % Replace with the correct path to your CSV file
data = readtable(file_path);

% Extract the X and Y columns (replace 'X' and 'Y' with the actual column names)
x = data{:, 1}; % Assuming the first column contains X values
y = data{:, 2}; % Assuming the second column contains Y values

% Define the number of bins
numBins = 256; % You can adjust the number of bins based on your data

% Generate the 2D histogram
figure;
histogram2(x, y, numBins, 'DisplayStyle', 'tile', 'ShowEmptyBins', 'on');

% Customize the plot
xlabel('X Values');
ylabel('Y Values');
zlabel('Frequency');
title('2D Histogram');
colorbar;

