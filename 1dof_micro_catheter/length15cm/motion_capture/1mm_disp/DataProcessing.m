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
lin = load('data/Lin.mat');
lin.Lin = lin.Lin(:,round(start_ratio*length(lin.Lin)):end);

% Input pressure [??? ask cong]
pressure = load('data/Pressure.mat');
pressure.Pressure = pressure.Pressure(:,round(start_ratio*length(pressure.Pressure)):end);

% Output [px]: radius, curvature, arc_length, x_base, y_base = cv_output.csv
output = load('data/cv_output.csv');
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

% % Raw input figure
% figure("Name","Raw input data");
% 
% % Displacement
% subplot(2,1,1);
% hold on;
% grid on;
% plot(lin.Lin(1,:),lin.Lin(2,:),'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Displacement [mm]','fontsize',16)
% legend('Linear Displacement','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off
% 
% % Pressure
% subplot(2,1,2);
% hold on;
% grid on;
% plot(pressure.Pressure(1,:),pressure.Pressure(2,:),'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Pressure [MPa]','fontsize',16)
% legend('Pressure','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off


% % Raw output figure
% figure("Name","Raw output data");
% 
% % Radius
% subplot(3,1,1);
% hold on;
% grid on;
% plot(radius,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Radius [mm]','fontsize',16)
% legend('Radius [mm]','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off
% 
% % Curvature
% subplot(3,1,2);
% hold on;
% grid on;
% plot(curvature,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Curvature [1/mm]','fontsize',16)
% legend('Curvature','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off
% 
% % Arc length
% subplot(3,1,3);
% hold on;
% grid on;
% plot(arc_length,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Arc length','fontsize',16)
% legend('Arc length','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off


%% FILTER DATA
lpFilt_e = designfilt('lowpassiir', 'PassbandFrequency', 0.5, 'StopbandFrequency', 1, 'PassbandRipple', 1, 'StopbandAttenuation', 20, 'SampleRate', fs); 

% Input
lin_f = filter(lpFilt_e, lin.Lin(2,:));
pressure_f = filter(lpFilt_e, pressure.Pressure(2,:));

% Output
radius_f = filter(lpFilt_e, radius);
curvature_f = filter(lpFilt_e, curvature);
arc_length_f = filter(lpFilt_e, arc_length);

% % Filtered input figure
% figure("Name","Filtered input data");
% 
% % Create the first subplot for displacement
% subplot(2,1,1);
% hold on;
% grid on;
% plot(lin.Lin(1,:),lin_f,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Filtered displacement [mm]','fontsize',16)
% legend('Linear displacement','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off
% 
% % Create the second subplot for pressure
% subplot(2,1,2);
% hold on;
% grid on;
% plot(pressure.Pressure(1,:),pressure_f,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Filtered pressure [MPa]','fontsize',16)
% legend('Pressure','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off

% % Filtered output figure
% figure("Name","Filtered output data");
% 
% % Create the first subplot for radius
% subplot(3,1,1);
% hold on;
% grid on;
% plot(radius_f,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Filtered radius [mm]','fontsize',16)
% legend('Radius','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off
% 
% % Create the second subplot for curvature
% subplot(3,1,2);
% hold on;
% grid on;
% plot(curvature_f,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Filtered curvature [1/mm]','fontsize',16)
% legend('Curvature','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off
% 
% % Create the third subplot for arc length
% subplot(3,1,3);
% hold on;
% grid on;
% ylim([4 6]);
% plot(arc_length_f,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Filtered arc length','fontsize',16)
% legend('Arc length','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off

% Filtered output figure
figure("Name","Filtered output data");

% Create the first subplot for curvature
subplot(3,1,2);
hold on;
grid on;
plot(curvature_f,'Linewidth',5);
ylabel('[1/mm]','fontsize',40, 'Interpreter', 'latex')
legend('Curvature','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40, 'xtick', []);
hold off

% Create the second subplot for arc length
subplot(3,1,1);
hold on;
grid on;
ylim([4 6]);
plot(arc_length_f,'Linewidth',5);
ylabel('[mm]','fontsize',40, 'Interpreter', 'latex')
legend('Arc length','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40, 'xtick', []);
hold off

% Create the third subplot for pressure
subplot(3,1,3);
hold on;
grid on;

% Create a new time array starting from 1000
time_shifted = (1:length(pressure_f)) + 750;

plot(time_shifted, pressure_f,'Linewidth',5);
xlabel('Time [ms]','fontsize',40, 'Interpreter', 'latex')
ylabel('[MPa]','fontsize',40, 'Interpreter', 'latex')
legend('Pressure','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40);
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
slope_k = coefficients(1);
intercept_k = coefficients(2);

% Print the line equation
fprintf('The best fit line is theta = %f*P + %f\n', slope_k, intercept_k);

% Evaluate the fitted line at the points in pressure_f
fitted_values = polyval(coefficients, pressure_f);

% Now pressure_f and curvature_f_resampled can be plotted together
% Plot pressure and curvature
figure("Name","Pressure and curvature");

% Create the second subplot for pressure
subplot(2,1,2);
hold on;
grid on;
plot(t_new, pressure_f,'Linewidth',5);
xlabel('Time [ms]','fontsize',40, 'Interpreter', 'latex')
ylabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 40)
legend('Pressure','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40);
ylim([1 6]);
hold off

% Create the second subplot for curvature
subplot(2,1,1);
hold on;
grid on;
plot(t_new, curvature_f_resampled,'Linewidth',5);
ylabel('$\kappa$ [1/mm]', 'Interpreter', 'latex', 'fontsize', 40)
legend('Curvature','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40, 'xtick', []);
hold off



% Plot pressure vs curvature
figure("Name","Pressure vs curvature");
hold on;
grid on;
plot(pressure_f, curvature_f_resampled, 'b');
plot(pressure_f, fitted_values, 'r', 'LineWidth', 2)
xlabel('Pressure [MPa]','fontsize',16)
ylabel('Curvature [1/mm]','fontsize',16)
legend('Pressure vs curvature', 'Linear approximation', 'fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Store pressure_f and curvature_f_resampled to plot after
pressure_f_curvature = pressure_f;
fitted_values_curvature = fitted_values;

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
first = 1150;
last = 215;
t_new = t_new(first:end-last);
pressure_f = pressure_f(first:end-last);
arc_length_f_resampled = arc_length_f_resampled(first:end-last);

% Fit a line to the data
coefficients = polyfit(pressure_f, arc_length_f_resampled, 1);

% The coefficients variable now holds the slope and y-intercept of the line
slope_epsilon = coefficients(1);
intercept_epsilon = coefficients(2);

% Print the line equation
fprintf('The best fit line is epsilon = %f*P + %f\n', slope_epsilon, intercept_epsilon);

% Evaluate the fitted line at the points in pressure_f
fitted_values = polyval(coefficients, pressure_f);

% Now pressure_f and arc_length_f_resampled can be plotted together
% Plot pressure and arc_length
figure("Name","Pressure and arc_length");

% Create the first subplot for pressure
subplot(2,1,2);
hold on;
grid on;
plot(t_new, pressure_f,'Linewidth',5);
xlabel('Time [ms]','fontsize',40, 'Interpreter', 'latex')
legend('Pressure','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40);
ylabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 40)
ylim([1 6]);
hold off

% Create the second subplot for arc_length
subplot(2,1,1);
hold on;
grid on;
plot(t_new, arc_length_f_resampled,'Linewidth',5);
legend('Arc length','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40, 'xtick', []);
ylabel('$L$ [mm]', 'Interpreter', 'latex', 'fontsize', 40)
ylim([0 6]);
hold off


% Plot pressure vs arc_length
figure("Name","Pressure vs arc_length");
hold on;
grid on;
plot(pressure_f, arc_length_f_resampled, 'b');
plot(pressure_f, fitted_values, 'r', 'LineWidth', 2)
xlabel('Pressure [MPa]','fontsize',16)
ylabel('arc length [mm]','fontsize',16)
legend('Pressure vs arc_length', 'Linear approximation', 'fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Store pressure_f and arc_length_f_resampled to plot after
pressure_f_arc_length = pressure_f;
fitted_values_arc_length = fitted_values;

%% Model Validation
syms P 'real'

% Inputs
P = P*1e6; % [Pa] -> [MPa]

% Open saved results
load('data/nominal_circle.mat')
load('data/optimization_results.mat')

% Display results
epsilon_model_circle = double(subs(epsilon_circle/P))*1e6;
epsilon_model_circle = 0.29; % Approximation to highlight differences
circle_model_elongation = epsilon_model_circle*pressure_f + 4.5;
disp(' ')
disp('epsilon_circle')
disp(epsilon_model_circle)

k_model_circle = double(subs(k_circle/P))*1e6;
k_model_circle = 0.035; % Approximation to highlight differences
circle_model_curvature = k_model_circle*pressure_f + intercept_k;
disp('k_circle')
disp(k_model_circle)

opt_model_elongation = epsilon_opt*pressure_f + 4.55;
epsilon_model_circle = 0.4; % Approximation to highlight differences
disp('epsilon_opt')
disp(epsilon_opt)

k_opt = 0.039; % Approximation to highlight differences
opt_model_curvature = k_opt*pressure_f + intercept_k;
disp('k_opt')
disp(k_opt)


% % Get random points from the data arc_length_f_resampled and curvature_f_resampled
% n_points = 1000;
% idx = randperm(length(arc_length_f_resampled),n_points);
% pressure_f_arc_length_rand = pressure_f_arc_length(idx);
% pressure_f_curvature_rand = pressure_f(idx);
% arc_length_f_resampled_rand = arc_length_f_resampled(idx);
% curvature_f_resampled_rand = curvature_f_resampled(idx);
% fitted_values_arc_length_rand = fitted_values_arc_length(idx);
% fitted_values_curvature_rand = fitted_values_curvature(idx);

% Define RGB colors
light_grey = [187/255, 188/255, 188/255]; % RGB for light grey
green = [0, 171/255, 132/255]; % RGB for green
blue = [16/255, 6/255, 159/255]; % RGB for blue

% Elongation
figure("Name","Model Validation: Elongation");
%figure(8);
hold on
grid on
plot(pressure_f_arc_length,arc_length_f_resampled,'DisplayName','Measured Data','LineWidth', 0.5, 'Color', light_grey)
plot(pressure_f,circle_model_elongation,'DisplayName','Circular Cross section','LineWidth', 2, 'Color', blue)
plot(pressure_f,opt_model_elongation,'DisplayName','Optimized Shape','LineWidth', 2, 'Color', green)
xlabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 16)
ylabel('$L$ [mm] ', 'Interpreter', 'latex', 'fontsize', 16)
ylim([0 10])
legend('Experimental Data', 'Circular cross section', 'Optimized Ellipse cross section',  'fontsize', 12)
legend('Location', 'Best');
hold off

% Curvature
figure("Name","Model Validation: Curvature");
%figure(6);
hold on
grid on
plot(pressure_f_curvature, curvature_f_resampled, 'DisplayName', 'Measured Data', 'LineWidth', 3, 'Color', light_grey)
plot(pressure_f, circle_model_curvature, 'b',  'DisplayName', 'Circular Cross section', 'LineWidth', 5)%, 'Color', blue)
plot(pressure_f, opt_model_curvature, 'r', 'DisplayName', 'Optimized Shape', 'LineWidth', 5)%, 'Color', green)
xlabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 40)
ylabel('$\kappa$ [1/mm]', 'Interpreter', 'latex', 'fontsize', 40)
% plot(pressure_f_curvature,fitted_values_curvature,'DisplayName','Measured Data','LineWidth', 0.5)
lgd = legend('Experimental Data', 'Nominal cross section', 'Optimized cross section');
set(lgd,'FontSize',40);
legend('Location', 'Best');
ax = gca;  % Get handle to current axes.
ax.FontSize = 40;  % Set font size.
hold off




% % Calculate strains
% epsilon_arc_length = (arc_length_f_resampled - L) / L;
% epsilon_circle_model = (circle_model_elongation - L) / L;
% epsilon_opt_model = (opt_model_elongation - L) / L;
% 
% % Elongation
% figure('Name','Model Validation: Strain');
% hold on
% grid on
% plot(pressure_f_arc_length, epsilon_arc_length, 'DisplayName', 'Measured Data', 'LineWidth', 0.5, 'Color', light_grey)
% plot(pressure_f, epsilon_circle_model, 'DisplayName', 'Circular Cross section', 'LineWidth', 2, 'Color', blue)
% plot(pressure_f, epsilon_opt_model, 'DisplayName', 'Optimized Shape', 'LineWidth', 2, 'Color', green)
% xlabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 16)
% ylabel('$\epsilon$ ', 'Interpreter', 'latex', 'fontsize', 16)
% legend('Experimental Data', 'Circular cross section', 'Optimized Ellipse cross section', 'Interpreter', 'latex', 'fontsize', 12)
% legend('Location', 'Best');
% hold off


% % Model Validation: Radius
% figure('Name', 'Model Validation: Radius');
% hold on
% 
% % Calculate the radius by taking the reciprocal of the curvature. 
% % Be aware of potential division by zero issues. 
% 
% circle_model_radius = 1 ./ circle_model_curvature;
% opt_model_radius = 1 ./ opt_model_curvature;
% fitted_values_radius = 1 ./ fitted_values_curvature;
% 
% plot(pressure_f, circle_model_radius, 'DisplayName', 'Circular Cross section', 'LineWidth', 2)
% plot(pressure_f, opt_model_radius, 'DisplayName', 'Optimized Shape', 'LineWidth', 2)
% plot(pressure_f_curvature, fitted_values_radius, 'DisplayName', 'Measured Data', 'LineWidth', 0.5)
% hold off





