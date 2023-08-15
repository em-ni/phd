clc;
clear;
close all;

% Input pressure [MPa]
pressure = load('data/Pressure.mat');

% Save as CSV for python processing
writematrix(pressure.Pressure, 'data/Pressure.csv');

% load optimization results
load('optimization_results.mat');

% Store epsilon_opt and k_opt into a matrix
optim_results = [epsilon_opt; k_opt];

% Save the matrix into a CSV file
writematrix(optim_results, 'data/optimization_results.csv');


