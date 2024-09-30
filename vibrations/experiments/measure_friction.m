clear all
close all
clc

% Open the saved data
load('trimmed_data.mat');

% take the integral of the data
integral_data = [];
for k = 1:16
    integral_data = [integral_data; cumtrapz(trimmed_data_list(k,:))];
end

% Take the integral of the wov data
integral_wov = cumtrapz(wov);

% Make a 3d plot with x axis the amplitude, y axis the frequency, and z axis the integral of the data (last point)
% Every triplet is taken as follows:
% Each files it's named as wv_A<amplitude>F<frequency> where amplitude is the amplitude of the wave and frequency is the frequency of the wave
% While the integral of the data is taken as the last point of the integral_data
% This si the order of the data in trimmed_data_list and so the order of the data in integral_data
% wv_A0.5F0.05.mat
% wv_A0.5F0.065.mat
% wv_A0.5F0.083.mat
% wv_A0.5F0.1.mat
% wv_A0.6F0.05.mat
% wv_A0.6F0.065.mat
% wv_A0.6F0.083.mat
% wv_A0.6F0.1.mat
% wv_A0.7F0.05.mat
% wv_A0.7F0.065.mat
% wv_A0.7F0.083.mat
% wv_A0.7F0.1.mat
% wv_A0.8F0.05.mat
% wv_A0.8F0.065.mat
% wv_A0.8F0.083.mat
% wv_A0.8F0.1.mat

amplitudes = [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8];
frequencies = [0.05, 0.065, 0.083, 0.1, 0.05, 0.065, 0.083, 0.1, 0.05, 0.065, 0.083, 0.1, 0.05, 0.065, 0.083, 0.1];
integrals = integral_data(:, end);

% interpolate the data
% Create a grid of amplitudes and frequencies
[amplitudes_grid, frequencies_grid] = meshgrid(0.5:0.01:0.8, 0.05:0.0015:0.1);

% Interpolate the integral data
integral_data_interp = griddata(amplitudes, frequencies, integrals, amplitudes_grid, frequencies_grid, 'natural');

% Do the same for thw wov data
full_integral_wov = [];
for k = 1:16
    full_integral_wov = [full_integral_wov; integral_wov];
end
full_integral_wov = full_integral_wov(:, end);
integral_wov_interp = griddata(amplitudes, frequencies, full_integral_wov, amplitudes_grid, frequencies_grid, 'cubic');

% Create a 3D surface plot
figure;
surf(amplitudes_grid, frequencies_grid, integral_data_interp);
hold on;
surf(amplitudes_grid, frequencies_grid, integral_wov_interp);
hold on;
% Plot the triplets as scatter points
% scatter3(amplitudes, frequencies, integrals, 100, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
xlabel('Amplitude');
ylabel('Frequency');
zlabel('Cumulative Friction');
title('Cumulative Friction vs. Amplitude and Frequency');
legend('Vibrations', 'No Vibrations');
xlim([0.4 0.9]);
ylim([0.04 0.11]);
hold off;


% Compute the ratio of the cumulative friction with and without vibrations
for k = 1:16
    ratio = 1 - integral_data(k, end) / integral_wov(end);
    fprintf('Amplitude: %.2f, Frequency: %.3f, Reduction Ratio: %.3f\n', amplitudes(k), frequencies(k), ratio);
end


% Unique amplitudes
unique_amps = unique(amplitudes);

% Plotting
figure;
hold on;
colors = lines(length(unique_amps)); % Get a colormap array
for i = 1:length(unique_amps)
    amp_mask = amplitudes == unique_amps(i);
    freqs_for_amp = frequencies(amp_mask);
    ints_for_amp = integrals(amp_mask);
    plot(freqs_for_amp, ints_for_amp, 'o-', 'Color', colors(i,:), 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('Amplitude = %.1f', unique_amps(i)));
end
xlabel('Frequency');
ylabel('Cumulative Friction');
title('Cumulative Friction vs. Frequency for Different Amplitudes');
legend('show');
grid on;
hold off;


% Unique frequencies
unique_freqs = unique(frequencies);

% Plotting
figure;
hold on;
colors = lines(length(unique_freqs)); % Get a colormap array
for i = 1:length(unique_freqs)
    freq_mask = frequencies == unique_freqs(i);
    amps_for_freq = amplitudes(freq_mask);
    ints_for_freq = integrals(freq_mask);
    plot(amps_for_freq, ints_for_freq, 'o-', 'Color', colors(i,:), 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('Frequency = %.3f', unique_freqs(i)));
end
xlabel('Amplitude');
ylabel('Cumulative Friction');
title('Cumulative Friction vs. Amplitude for Different Frequencies');
legend('show');
grid on;
hold off;