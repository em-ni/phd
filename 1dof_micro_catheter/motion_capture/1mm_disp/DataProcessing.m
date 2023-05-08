%% Setup
clc;
clear;
close all;

% Sampling frequency (input)
fs = 200;
Ts = 1/fs; 

% Tip length
L = 5; % [mm]

% Starting time of the experiment
% Mouse clicked at 22 [s]
video_duration = 82; % [s] 
start_ratio = 22/video_duration;

%% LOAD DATA
% Input displacement [mm? ask cong]
lin = load('Lin.mat');
lin.Lin = lin.Lin(:,round(start_ratio*length(lin.Lin)):end);

% Input pressure [??? ask cong]
pressure = load('Pressure.mat');
pressure.Pressure = pressure.Pressure(:,round(start_ratio*length(pressure.Pressure)):end);

% Output [px]: radius, curvature, arc_length = curvature.csv
output = load('curvature.csv');
radius = output(:,1);
radius = radius(round(start_ratio*length(radius)):end);

%curvature = output(:,2);
arc_length = output(:,3);
arc_length = arc_length(round(start_ratio*length(arc_length)):end);

% Compute conversion rate from px to mm
% Get average of the first 1250 points of arc_length
arc_length_avg = mean(arc_length(1:1250));

% Compute conversion rate
conv_rate = L/arc_length_avg;

% Convert radius, curvature and arc_length to mm
radius = radius*conv_rate;
curvature = 1 ./ radius;
arc_length = arc_length*conv_rate;

% Raw input figure
figure("Name","Raw input data");

% Displacement
subplot(2,1,1);
hold on;
grid on;
plot(lin.Lin(1,:),lin.Lin(2,:),'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Displacement','fontsize',16)
legend('Linear Displacement','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Pressure
subplot(2,1,2);
hold on;
grid on;
plot(pressure.Pressure(1,:),pressure.Pressure(2,:),'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Pressure','fontsize',16)
legend('Pressure','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off


% Raw output figure
figure("Name","Raw output data");

% Radius
subplot(3,1,1);
hold on;
grid on;
plot(radius,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Radius','fontsize',16)
legend('Radius','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Curvature
subplot(3,1,2);
hold on;
grid on;
plot(curvature,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Curvature','fontsize',16)
legend('Curvature','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Arc length
subplot(3,1,3);
hold on;
grid on;
plot(arc_length,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Arc length','fontsize',16)
legend('Arc length','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off


%% FILTER DATA
lpFilt_e = designfilt('lowpassiir', 'PassbandFrequency', 0.5, 'StopbandFrequency', 1, 'PassbandRipple', 1, 'StopbandAttenuation', 20, 'SampleRate', fs); 

% Input
lin_f = filter(lpFilt_e, lin.Lin(2,:));
pressure_f = filter(lpFilt_e, pressure.Pressure(2,:));

% Output
radius_f = filter(lpFilt_e, radius);
curvature_f = filter(lpFilt_e, curvature);
arc_length_f = filter(lpFilt_e, arc_length);

% Filtered input figure
figure("Name","Filtered input data");

% Create the first subplot for displacement
subplot(2,1,1);
hold on;
grid on;
plot(lin.Lin(1,:),lin_f,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered displacement','fontsize',16)
legend('Linear displacement','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for pressure
subplot(2,1,2);
hold on;
grid on;
plot(pressure.Pressure(1,:),pressure_f,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered pressure','fontsize',16)
legend('Pressure','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Filtered output figure
figure("Name","Filtered output data");

% Create the first subplot for radius
subplot(3,1,1);
hold on;
grid on;
plot(radius_f,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered radius','fontsize',16)
legend('Radius','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for curvature
subplot(3,1,2);
hold on;
grid on;
plot(curvature_f,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered curvature','fontsize',16)
legend('Curvature','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the third subplot for arc length
subplot(3,1,3);
hold on;
grid on;
ylim([4 6]);
plot(arc_length_f,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered arc length','fontsize',16)
legend('Arc length','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

%% Resamplign data
% Assuming t_pressure and t_curvature are the time vectors for pressure_f and curvature_f

% Make sure t_pressure and t_curvature cover the same time span
t_pressure = 1:1:length(pressure.Pressure(1,:));
t_curvature = 1:1:length(curvature_f);
t_min = max(min(t_pressure), min(t_curvature));
t_max = min(max(t_pressure), max(t_curvature));

% Create new time vector with the same range but matching the length of pressure_f
t_new = linspace(t_min, t_max, length(pressure_f));

% Interpolate curvature_f to the new times
curvature_f_resampled = interp1(t_curvature, curvature_f, t_new);

% Get peaks 
[pks_p,locs_p] = findpeaks(pressure_f);
[pks_c,locs_c] = findpeaks(curvature_f_resampled);

% Get first peaks locations
first_p = locs_p(1);
first_c = locs_c(1);

% Display the result
disp("First peak of pressure_f is at " + first_p + " and first peak of curvature_f_resampled is at " + first_c);

% Shift curvature_f_resampled to match the first peak of pressure_f
curvature_f_resampled = circshift(curvature_f_resampled, first_p - first_c);

% Clean data by removing the first 100 points and the last 500 points
t_new = t_new(1000:end-2000);
pressure_f = pressure_f(1000:end-2000);
curvature_f_resampled = curvature_f_resampled(1000:end-2000);

% Fit a line to the data
coefficients = polyfit(pressure_f, curvature_f_resampled, 1);

% The coefficients variable now holds the slope and y-intercept of the line
slope = coefficients(1);
intercept = coefficients(2);

% Print the line equation
fprintf('The best fit line is theta = %f*P + %f\n', slope, intercept);

% Evaluate the fitted line at the points in pressure_f
fitted_values = polyval(coefficients, pressure_f);

% Now pressure_f and curvature_f_resampled can be plotted together
% Plot pressure and curvature
figure("Name","Pressure and curvature");

% Create the second subplot for pressure
subplot(2,1,1);
hold on;
grid on;
plot(t_new, pressure_f,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered pressure','fontsize',16)
legend('Pressure','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for curvature
subplot(2,1,2);
hold on;
grid on;
plot(t_new, curvature_f_resampled,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered curvature','fontsize',16)
legend('Curvature','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Plot pressure vs curvature
figure("Name","Pressure vs curvature");
hold on;
grid on;
plot(pressure_f, curvature_f_resampled, 'b');
plot(pressure_f, fitted_values, 'r', 'LineWidth', 2)
xlabel('Pressure','fontsize',16)
ylabel('Curvature','fontsize',16)
legend('Pressure vs curvature', 'Linear approximation', 'fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Arc length
%% Resamplign data
% Assuming t_pressure and t_arc_length are the time vectors for pressure_f and arc_length_f

% Make sure t_pressure and t_arc_length cover the same time span
t_pressure = 1:1:length(pressure.Pressure(1,:));
t_arc_length = 1:1:length(arc_length_f);
t_min = max(min(t_pressure), min(t_arc_length));
t_max = min(max(t_pressure), max(t_arc_length));

% Create new time vector with the same range but matching the length of pressure_f
t_new = linspace(t_min, t_max, length(pressure_f));

% Interpolate arc_length_f to the new times
arc_length_f_resampled = interp1(t_arc_length, arc_length_f, t_new);

% Get peaks 
[pks_p,locs_p] = findpeaks(pressure_f);
[pks_c,locs_c] = findpeaks(arc_length_f_resampled);

% Get first peaks locations
first_p = locs_p(1);
first_c = locs_c(1);

% Display the result
disp("First peak of pressure_f is at " + first_p + " and first peak of arc_length_f_resampled is at " + first_c);

% Shift arc_length_f_resampled to match the first peak of pressure_f
arc_length_f_resampled = circshift(arc_length_f_resampled, first_p - first_c);

% Clean data by removing the first 100 points and the last 500 points
first = 100;
last = 500;
t_new = t_new(first:end-last);
pressure_f = pressure_f(first:end-last);
arc_length_f_resampled = arc_length_f_resampled(first:end-last);

% Fit a line to the data
coefficients = polyfit(pressure_f, arc_length_f_resampled, 1);

% The coefficients variable now holds the slope and y-intercept of the line
slope = coefficients(1);
intercept = coefficients(2);

% Print the line equation
fprintf('The best fit line is epsilon = %f*P + %f\n', slope, intercept);

% Evaluate the fitted line at the points in pressure_f
fitted_values = polyval(coefficients, pressure_f);

% Now pressure_f and arc_length_f_resampled can be plotted together
% Plot pressure and arc_length
figure("Name","Pressure and arc_length");

% Create the second subplot for pressure
subplot(2,1,1);
hold on;
grid on;
plot(t_new, pressure_f,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered pressure','fontsize',16)
legend('Pressure','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for arc_length
subplot(2,1,2);
hold on;
grid on;
plot(t_new, arc_length_f_resampled,'Linewidth',2);
xlabel('Time','fontsize',16)
ylabel('Filtered arc_length','fontsize',16)
legend('arc_length','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Plot pressure vs arc_length
figure("Name","Pressure vs arc_length");
hold on;
grid on;
plot(pressure_f, arc_length_f_resampled, 'b');
plot(pressure_f, fitted_values, 'r', 'LineWidth', 2)
xlabel('Pressure','fontsize',16)
ylabel('arc_length','fontsize',16)
legend('Pressure vs arc_length', 'Linear approximation', 'fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off








