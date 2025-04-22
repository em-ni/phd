% A range 0.5 0.6 0.7 0.8
% F range 0.05  0.065 0.083  0.1

clear all
close all
clc

%% Load, filter and plot the data
% Define the window size for the moving average filter
filterWindowSize = 100;
b = (1/filterWindowSize)*ones(1,filterWindowSize);
a = 1;

wov_full = load('wov.mat');
wov = -wov_full.Loadcell(2,:);

% Filter the wov data
filtered_wov = filter(b, a, wov);

figure
subplot(1, 2, 1);
hold on
title('No Vibrations')
plot(wov)
hold off

subplot(1, 2, 2);
hold on
title('Filtered No Vibrations')
plot(filtered_wov)
hold off


figure
hold on
% Folder where the mat files are stored
folder = './grid_data'; % Update this with the path to your .mat files

% List all .mat files in the folder except for the last two
files = dir(fullfile(folder, 'wv_*.mat'));

% Check to only plot up to 16 files, including 'wov.mat'
max_plots = 17;

% Loop through each file and plot data
for k = 2:max_plots
    full_path = fullfile(files(k-1).folder, files(k-1).name);
    data = load(full_path);
    subplot(4,4,k-1);
    plot(-data.Loadcell(2,:)); % Assuming data structure is similar
    modified_title = strrep(files(k-1).name, 'wv_', ''); % Remove 'wv_' from the title
    title(modified_title, 'Interpreter', 'none'); % 'none' to correctly display underscores in titles
end

% Adjust subplot spacing
sgtitle('Vibrations Experiment: Raw Data'); % Super title for all subplots
hold off;

% % Create a new figure for the filtered data
% figure;
% hold on;
% sgtitle('Vibrations Experiment: Filtered Data');

% % Process each file and plot the filtered data
% for k = 1:min(length(files), 16)
%     full_path = fullfile(files(k).folder, files(k).name);
%     data = load(full_path);
%     filtered_data = filter(b, a, -data.Loadcell(2,:)); % Apply the moving average filter
    
%     subplot(4,4,k);
%     plot(filtered_data);
%     hold on; % Keep the subplot active
%     plot(filtered_wov, 'r--', 'LineWidth', 1); % Overlay the filtered 'wov' data
%     hold off; % Release the subplot
%     modified_title = strrep(files(k).name, 'wv_', ''); % Remove 'wv_' from the title
%     title(modified_title, 'Interpreter', 'none'); % Correctly display titles
% end

% hold off;

%% Uniform data by cutting a fixed window size
% Define the window size for the uniform data
experimentfilterWindowSize = 10000;

% define the starting point for each experiment
wov_start = 0.45*1e4;

A05F005_start = 0.5*1e4;
A05F065_start = 0.5*1e4;
A05F083_start = 0.5*1e4;
A05F1_start = 0.78*1e4;

A06F005_start = 0.4*1e4;
A06F065_start = 0.4*1e4;
A06F083_start = 0.4*1e4;
A06F1_start = 0.4*1e4;

A07F005_start = 0.4*1e4;
A07F065_start = 0.4*1e4;
A07F083_start = 0.4*1e4;
A07F1_start = 0.4*1e4;

A08F005_start = 0.4*1e4;
A08F065_start = 0.4*1e4;
A08F083_start = 0.4*1e4;
A08F1_start = 0.4*1e4;

% Trim wov
wov = filtered_wov(wov_start:wov_start+experimentfilterWindowSize);

% Data list
trimmed_data_list = [];

% Trim the data
for k = 1:min(length(files), 16)
    full_path = fullfile(files(k).folder, files(k).name);
    data = load(full_path);
    filtered_data = filter(b, a, -data.Loadcell(2,:)); % Apply the moving average filter
    name = files(k).name;
    disp(name)
    
    % Trim the data
    switch name
        case 'wv_A0.5F0.05.mat'
            A05F005 = filtered_data(A05F005_start:A05F005_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A05F005];
        case 'wv_A0.5F0.065.mat'
            A05F065 = filtered_data(A05F065_start:A05F065_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A05F065];
        case 'wv_A0.5F0.083.mat'
            A05F083 = filtered_data(A05F083_start:A05F083_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A05F083];
        case 'wv_A0.5F0.1.mat'
            A05F1 = filtered_data(A05F1_start:A05F1_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A05F1];
        case 'wv_A0.6F0.05.mat'
            A06F005 = filtered_data(A06F005_start:A06F005_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A06F005];
        case 'wv_A0.6F0.065.mat'
            A06F065 = filtered_data(A06F065_start:A06F065_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A06F065];
        case 'wv_A0.6F0.083.mat'
            A06F083 = filtered_data(A06F083_start:A06F083_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A06F083];
        case 'wv_A0.6F0.1.mat'
            A06F1 = filtered_data(A06F1_start:A06F1_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A06F1];
        case 'wv_A0.7F0.05.mat'
            A07F005 = filtered_data(A07F005_start:A07F005_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A07F005];
        case 'wv_A0.7F0.065.mat'
            A07F065 = filtered_data(A07F065_start:A07F065_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A07F065];
        case 'wv_A0.7F0.083.mat'
            A07F083 = filtered_data(A07F083_start:A07F083_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A07F083];
        case 'wv_A0.7F0.1.mat'
            A07F1 = filtered_data(A07F1_start:A07F1_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A07F1];
        case 'wv_A0.8F0.05.mat'
            A08F005 = filtered_data(A08F005_start:A08F005_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A08F005];
        case 'wv_A0.8F0.065.mat'
            A08F065 = filtered_data(A08F065_start:A08F065_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A08F065];
        case 'wv_A0.8F0.083.mat'
            A08F083 = filtered_data(A08F083_start:A08F083_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A08F083];
        case 'wv_A0.8F0.1.mat'
            A08F1 = filtered_data(A08F1_start:A08F1_start+experimentfilterWindowSize);
            trimmed_data_list = [trimmed_data_list; A08F1];
    end
end

% Plot the trimmed data
figure;
hold on;
sgtitle('Vibrations Experiment: Trimmed Data');
for k = 1:min(length(files), 16)
    subplot(4,4,k);
    plot(trimmed_data_list(k,:));
    hold on;
    plot(wov, 'r--', 'LineWidth', 1);
    xlim([0 10000]); % Set x-axis limits
    hold off;
    modified_title = strrep(files(k).name, 'wv_', ''); % Remove 'wv_' from the title
    title(modified_title, 'Interpreter', 'none'); % Correctly display titles
end

% Save the trimmed data
save('trimmed_data.mat', 'trimmed_data_list', 'wov');