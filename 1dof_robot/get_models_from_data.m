% Data processing to get the empirical linear model for pressure vs curvature fo the free motion experiment
% Setup
clc;
clear all;
close all;

% Tip length
L = 8.5*1e-3; % [m]

% Get data from csv file
data = load("data/free_motion/cv_output.csv");
pressure_bar = data(:,1); % [Bar]  divide by 10 to get MPa
pressure_mpa = pressure_bar / 10; % [MPa]
pressure = pressure_mpa * 1e6; % [Pa]
radius = data(:,3); % [px]
arc_length = data(:,5); % [px]

% Get conversion rate from px to m
arc_length_avg = mean(arc_length());
conv_rate = L/arc_length_avg; % [m/px]

% Convert to m
arc_length = arc_length*conv_rate; % [m]
radius = radius*conv_rate; % [m]
curvature = 1 ./ radius; % [1/m]
disp("Data loaded");

% In order to capture the minimum pressure necessary to bend the robot (the point when the input torque u is greater than the elastic term K)
% the model is fit only on the curvature data that is greater than a threshold
threshold = 30; % [1/m]
idx = curvature > threshold;
pressure_fit = pressure(idx);
curvature_fit = curvature(idx);

% Also set to zero only the values of curvature that are less than the threshold
curvature(curvature < threshold) = 0;

% Fit linear model
mdl = fitlm(pressure_fit, curvature_fit);
disp(mdl);

% Save linear model
save("data/free_motion/linear_model_p_vs_k.mat", "mdl");

% Get linear model parameters
k_data = mdl.Coefficients.Estimate(2);
k_intercept = mdl.Coefficients.Estimate(1);

% Plot linear model and original data points
figure;
plot(mdl, LineWidth=5)
hold on;
scatter(pressure, curvature, 'filled', LineWidth=5)
x = linspace(0, max(pressure)+0.1*max(pressure), 100);
y = k_data * x + k_intercept;
plot(x, y, 'r', LineWidth=5)
hold off;
ylabel("Curvature [1/m]", 'Interpreter', 'latex', 'fontsize', 30);
xlabel("Pressure [Pa]", 'Interpreter', 'latex', 'fontsize', 30);
title("Linear Model for Pressure vs Curvature for Free Motion Experiment", 'Interpreter', 'latex', 'fontsize', 30);
xlim([0, max(pressure)+0.1*max(pressure)]);
ylim([0, max(curvature)+0.1*max(curvature)]);
% Adjust x-ticks and x-tick labels for pressure
xticks_current = get(gca, 'xtick');
set(gca, 'xtick', xticks_current);
set(gca, 'xticklabel', xticks_current);
ax = gca;  % Get handle to current axes.
ax.FontSize = 30;  % Set font size.

% Same but for pressure vs epsilon
% Calculate epsilon
L0 = min(arc_length); % [m] 
epsilon = (arc_length - L0) / L0;
epsilon_fit = epsilon(idx);

% Fit linear model
mdl = fitlm(pressure_fit, epsilon_fit);
disp(mdl);

% Save linear model
save("data/free_motion/linear_model_p_vs_epsilon.mat", "mdl");

% Get linear model parameters
epsilon_data = mdl.Coefficients.Estimate(2);
epsilon_intercept = mdl.Coefficients.Estimate(1);

% Plot linear model and original data points
figure;
plot(mdl);
hold on;
scatter(pressure, epsilon, 'filled');
x = linspace(0, max(pressure)+0.1*max(pressure), 100);
y = epsilon_data * x + epsilon_intercept;
plot(x, y, 'r');
hold off;
ylabel("Epsilon");
xlabel("Pressure [Pa]");
title("Linear Model for Pressure vs Epsilon for Free Motion Experiment");
xlim([0, max(pressure)+0.1*max(pressure)]);
ylim([0, max(epsilon)+0.1*max(epsilon)]);
disp("Data processed");



