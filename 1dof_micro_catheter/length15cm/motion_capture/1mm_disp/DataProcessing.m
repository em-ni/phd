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
% grid off;
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
% grid off;
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
% grid off;
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
% grid off;
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
% grid off;
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
% grid off;
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
% grid off;
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
% grid off;
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
% grid off;
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
% grid off;
% ylim([4 6]);
% plot(arc_length_f,'Linewidth',2);
% xlabel('Time [s]','fontsize',16)
% ylabel('Filtered arc length','fontsize',16)
% legend('Arc length','fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off

% % Filtered output figure
% figure("Name","Filtered output data");
% 
% % Create the first subplot for curvature
% subplot(3,1,2);
% hold on;
% grid off;
% plot(curvature_f,'Linewidth',5);
% ylabel('[1/mm]','fontsize',40, 'Interpreter', 'latex')
% legend('Curvature','fontsize',40, 'Interpreter', 'latex')
% legend('Location', 'Best');
% set(gca,'FontSize',40, 'xtick', []);
% hold off
% 
% % Create the second subplot for arc length
% subplot(3,1,1);
% hold on;
% grid off;
% ylim([4 6]);
% plot(arc_length_f,'Linewidth',5);
% ylabel('[mm]','fontsize',40, 'Interpreter', 'latex')
% legend('Arc length','fontsize',40, 'Interpreter', 'latex')
% legend('Location', 'Best');
% set(gca,'FontSize',40, 'xtick', []);
% hold off
% 
% % Create the third subplot for pressure
% subplot(3,1,3);
% hold on;
% grid off;
% 
% % Create a new time array starting from 1000
% time_shifted = (1:length(pressure_f)) + 750;
% 
% plot(time_shifted, pressure_f,'Linewidth',5);
% xlabel('Time [ms]','fontsize',40, 'Interpreter', 'latex')
% ylabel('[MPa]','fontsize',40, 'Interpreter', 'latex')
% legend('Pressure','fontsize',40, 'Interpreter', 'latex')
% legend('Location', 'Best');
% set(gca,'FontSize',40);
% hold off





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

% Create the subplot for pressure
subplot(2,1,2);
hold on;
% grid off;
plot(t_new, pressure_f,'Linewidth',5);
xlabel('Time [ms]','fontsize',40, 'Interpreter', 'latex')

% Adjust y-ticks and y-tick labels for pressure
yticks_current = get(gca, 'ytick');
set(gca, 'ytick', yticks_current);
set(gca, 'yticklabel', yticks_current / 10);

ylabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 40)
legend('Pressure','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40);
ylim([1 6]);


% Create the subplot for curvature
subplot(2,1,1);
hold on;
% grid off;
plot(t_new, curvature_f_resampled,'Linewidth',5);
ylabel('$\kappa$ [1/mm]', 'Interpreter', 'latex', 'fontsize', 40)
legend('Curvature','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40, 'xtick', []);





% % Plot pressure vs curvature
% figure("Name","Pressure vs curvature");
% hold on;
% grid off;
% plot(pressure_f, curvature_f_resampled, 'b');
% plot(pressure_f, fitted_values, 'r', 'LineWidth', 2)
% xlabel('Pressure [MPa]','fontsize',16)
% ylabel('Curvature [1/mm]','fontsize',16)
% legend('Pressure vs curvature', 'Linear approximation', 'fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off

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
% grid off;
plot(t_new, pressure_f,'Linewidth',5);
xlabel('Time [ms]','fontsize',40, 'Interpreter', 'latex')

% Adjust y-ticks and y-tick labels for pressure
yticks_current = get(gca, 'ytick');
set(gca, 'ytick', yticks_current);
set(gca, 'yticklabel', yticks_current / 10);

ylabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 40)
legend('Pressure','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40);
ylim([1 6]);
hold off

% Create the second subplot for arc_length
subplot(2,1,1);
hold on;
% grid off;
plot(t_new, arc_length_f_resampled,'Linewidth',5);
legend('Arc length','fontsize',40, 'Interpreter', 'latex')
legend('Location', 'Best');
set(gca,'FontSize',40, 'xtick', []);

% Setting more detailed y-ticks
yticks(4:0.5:6);  % Set y-ticks from 4 to 6 with 0.1 increments
ylim([4 6]);

ylabel('$L$ [mm]', 'Interpreter', 'latex', 'fontsize', 40)
hold off




% % Plot pressure vs arc_length
% figure("Name","Pressure vs arc_length");
% hold on;
% grid off;
% plot(pressure_f, arc_length_f_resampled, 'b');
% plot(pressure_f, fitted_values, 'r', 'LineWidth', 2)
% xlabel('Pressure [MPa]','fontsize',16)
% ylabel('arc length [mm]','fontsize',16)
% legend('Pressure vs arc_length', 'Linear approximation', 'fontsize',16)
% legend('Location', 'Best');
% %set(gca,'FontSize',20);
% hold off

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
% light_grey = [187/255, 188/255, 188/255]; % RGB for light grey
% green = [0, 171/255, 132/255]; % RGB for green
% blue = [16/255, 6/255, 159/255]; % RGB for blue

% Using the same colors and line widths as the second plot
light_grey = [0.7, 0.7, 0.7]; % Assuming this color value for light_grey
blue = [0, 0, 1];  % Default blue color in MATLAB
green = [0, 1, 0]; % Assuming this color value for green

% % Elongation
% figure("Name","Model Validation: Elongation");
% %figure(8);
% hold on
% grid off
% plot(pressure_f_arc_length,arc_length_f_resampled,'DisplayName','Measured Data','LineWidth', 0.5, 'Color', light_grey)
% plot(pressure_f,circle_model_elongation,'DisplayName','Circular Cross section','LineWidth', 2, 'Color', blue)
% plot(pressure_f,opt_model_elongation,'DisplayName','Optimized Shape','LineWidth', 2, 'Color', green)
% xlabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 16)
% ylabel('$L$ [mm] ', 'Interpreter', 'latex', 'fontsize', 16)
% ylim([0 10])
% legend('Experimental Data', 'Circular cross section', 'Optimized Ellipse cross section',  'fontsize', 12)
% legend('Location', 'Best');
% hold off

figure("Name","Model Validation: Curvature");
hold on
grid off
% Determine unique pressure_f values and preallocate arrays for bounds
rounded_pressures_curvature = round(pressure_f_curvature, 1);
unique_pressures_curvature = unique(rounded_pressures_curvature);
curvature_upper = zeros(size(unique_pressures_curvature));
curvature_lower = zeros(size(unique_pressures_curvature));

% Calculate upper and lower bounds for each unique pressure_f value
for i = 1:length(unique_pressures_curvature)
    idx = rounded_pressures_curvature == unique_pressures_curvature(i);
    curvature_upper(i) = max(curvature_f_resampled(idx));
    curvature_lower(i) = min(curvature_f_resampled(idx));
end

% Create x and y vectors for fill
X_fill_curvature = [unique_pressures_curvature, fliplr(unique_pressures_curvature)]; 
Y_fill_curvature = [curvature_upper, fliplr(curvature_lower)];

% Use fill function to create shaded region
fill(X_fill_curvature, Y_fill_curvature, light_grey, 'EdgeColor', 'none', 'DisplayName', 'Experimental Data', 'FaceAlpha', 0.5);

% plot_k_interval = [1:10:length(pressure_f_curvature)-500];
% plot(pressure_f_curvature(plot_k_interval), curvature_f_resampled(plot_k_interval), 'DisplayName', 'Measured Data', 'LineWidth', 3, 'Color', light_grey)%, 'Marker', '*', 'MarkerSize', 10, 'LineStyle', 'none')
plot(pressure_f, circle_model_curvature, 'b',  'DisplayName', 'Circular Cross section', 'LineWidth', 5)
plot(pressure_f, opt_model_curvature, 'r', 'DisplayName', 'Optimized Shape', 'LineWidth', 5)
xlabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 40)
ylabel('$\kappa$ [1/mm]', 'Interpreter', 'latex', 'fontsize', 40)

% Adjust x-ticks and x-tick labels for pressure
xticks_current = get(gca, 'xtick');
set(gca, 'xtick', xticks_current);
set(gca, 'xticklabel', xticks_current / 10);

% Set y-axis limits to focus on specific range
ylim([0.025 0.175]);

lgd = legend('Experimental Data', 'Nominal cross section', 'Optimized cross section', 'Box', 'off');
set(lgd,'FontSize',40);
legend('Location', 'Best');
ax = gca;  % Get handle to current axes.
ax.FontSize = 40;  % Set font size.
hold off


% Redefine curvature_f_resampled as a straight line with slope 0.0388
curvature_f_resampled_linear = slope_k * pressure_f + intercept_k;

% Compute NRMSE for circle_model_curvature_resampled wrt curvature_f_resampled_linear
residuals_circle = circle_model_curvature - curvature_f_resampled_linear;
NRMSE_circle = sqrt(mean(residuals_circle.^2)) / (max(curvature_f_resampled_linear) - min(curvature_f_resampled_linear));
disp(['NRMSE for circle_model_curvature: ', num2str(NRMSE_circle)])

% Compute NRMSE for opt_model_curvature_resampled wrt curvature_f_resampled_linear
residuals_opt = opt_model_curvature - curvature_f_resampled_linear;
NRMSE_opt = sqrt(mean(residuals_opt.^2)) / (max(curvature_f_resampled_linear) - min(curvature_f_resampled_linear));
disp(['NRMSE for opt_model_curvature: ', num2str(NRMSE_opt)])


%% Apply kinematic model to the data
% Define old and new indices
old_indices = linspace(1, length(curvature_f_resampled), length(curvature_f_resampled));
new_indices = linspace(1, length(curvature_f_resampled), length(arc_length_f_resampled));

% Interpolate curvature_f_resampled to the desired length
curvature_f_resampled = interp1(old_indices, curvature_f_resampled, new_indices, 'linear');

% Reduce dimension of all vectors
% interval = [420:1900]; %orientation
interval = [200:2900];
pressure_f = pressure_f(interval);
curvature_f_resampled = curvature_f_resampled(interval);
arc_length_f_resampled = arc_length_f_resampled(interval);
circle_model_curvature = circle_model_curvature(interval);
opt_model_curvature = opt_model_curvature(interval);
circle_model_elongation = circle_model_elongation(interval);
opt_model_elongation = opt_model_elongation(interval);


% Define the kinematic model
x_t = arc_length_f_resampled .* ( 1 - cos(curvature_f_resampled.*arc_length_f_resampled)) ./ (curvature_f_resampled.*arc_length_f_resampled);
y_t = arc_length_f_resampled .* sin(curvature_f_resampled.*arc_length_f_resampled) ./ (curvature_f_resampled.*arc_length_f_resampled);

x_t_nominal = L .* ( 1 - cos(circle_model_curvature.*L)) ./ (circle_model_curvature.*L);
y_t_nominal = L .* sin(circle_model_curvature.*L) ./ (circle_model_curvature.*L)-0.01;

x_t_opt = L .* ( 1 - cos(opt_model_curvature.*L)) ./ (opt_model_curvature.*L);
y_t_opt = L .* sin(opt_model_curvature.*L) ./ (opt_model_curvature.*L);

alpha = curvature_f_resampled .* arc_length_f_resampled;
alpha_nominal = circle_model_curvature .* L;
alpha_opt = opt_model_curvature .* L;

% Plot the results
figure("Name","Kinematic Model: Position");
hold on
grid off

% Determine unique x_t values and preallocate arrays for bounds
rounded_xt = round(x_t,2);
unique_xt = unique(rounded_xt);
y_upper = zeros(size(unique_xt));
y_lower = zeros(size(unique_xt));

% Calculate upper and lower bounds for each unique x_t value
for i = 1:length(unique_xt)
    idx = rounded_xt == unique_xt(i);
    y_upper(i) = max(y_t(idx));
    y_lower(i) = min(y_t(idx));
end

% Create x and y vectors for fill
X_fill_position = [unique_xt, fliplr(unique_xt)]; 
Y_fill_position = [y_upper, fliplr(y_lower)];

% Use fill function to create shaded region
fill(X_fill_position, Y_fill_position, light_grey, 'EdgeColor', 'none', 'DisplayName', 'Experimental Data', 'FaceAlpha', 0.5);

%plot(x_t, y_t, 'DisplayName', 'Measured Data', 'LineWidth', 3, 'Color', light_grey, 'Marker', '*', 'MarkerSize', 10)
% plot_pos_interval = [1:1:length(x_t)];
% plot(x_t(plot_pos_interval), y_t(plot_pos_interval), 'DisplayName', 'Measured Data', 'LineWidth', 3, 'Color', light_grey)%, 'Marker', '*', 'MarkerSize', 10, 'LineStyle', 'none')

plot(x_t_nominal, y_t_nominal, 'b',  'DisplayName', 'Circular Cross section', 'LineWidth', 5)
plot(x_t_opt, y_t_opt, 'r', 'DisplayName', 'Optimized Shape', 'LineWidth', 5)
xlabel('$x_t$ [mm]', 'Interpreter', 'latex', 'fontsize', 40)
ylabel('$y_t$ [mm] ', 'Interpreter', 'latex', 'fontsize', 40)

lgd = legend('Experimental Data', 'Circular cross section', 'Optimized cross section', 'Interpreter', 'latex', 'fontsize', 40, 'Box', 'off');
set(lgd,'FontSize',40);
legend('Location', 'Best');
xlim([0.4 2])
ylim([4.35 5.25])

ax = gca;  % Get handle to current axes.
ax.FontSize = 40;  % Set font size.

hold off

figure("Name","Kinematic Model: Orientation");
hold on
grid off

% Determine unique pressure values and preallocate arrays for bounds
rounded_pressures = round(pressure_f, 1);
unique_pressures = unique(rounded_pressures);
alpha_upper = zeros(size(unique_pressures));
alpha_lower = zeros(size(unique_pressures));

% Calculate upper and lower bounds for each unique pressure value
for i = 1:length(unique_pressures)
    idx = rounded_pressures == unique_pressures(i);
    alpha_upper(i) = max(alpha(idx));
    alpha_lower(i) = min(alpha(idx));
end

% Create x and y vectors for fill
X_fill = [unique_pressures, fliplr(unique_pressures)]; 
Y_fill = [alpha_upper, fliplr(alpha_lower)];

% Use fill function to create shaded region
fill(X_fill, Y_fill, light_grey, 'EdgeColor', 'none', 'DisplayName', 'Experimental Data', 'FaceAlpha', 0.5);

% plot_or_interval = [1:10:length(pressure_f)-500];
% plot(pressure_f(plot_or_interval), alpha(plot_or_interval), 'DisplayName', 'Measured Data', 'LineWidth', 3, 'Color', light_grey)%, 'Marker', '*', 'MarkerSize', 10, 'LineStyle', 'none')

plot(pressure_f, alpha_nominal, 'b', 'DisplayName', 'Circular Cross section', 'LineWidth', 5);
plot(pressure_f, alpha_opt, 'r', 'DisplayName', 'Optimized Shape', 'LineWidth', 5);
xlabel('$p$ [MPa]', 'Interpreter', 'latex', 'fontsize', 40);
ylabel('$\theta$ [rad]', 'Interpreter', 'latex', 'fontsize', 40);

% Adjust x-ticks and x-tick labels for pressure
xticks_current = get(gca, 'xtick');
set(gca, 'xtick', xticks_current);
set(gca, 'xticklabel', xticks_current / 10);

% Set y-axis limits to focus on specific range
% xlim([1.5 6]);
% ylim([0 1]);

lgd = legend('Experimental Data', 'Nominal cross section', 'Optimized cross section', 'Box', 'off');
set(lgd,'FontSize',40);

legend('Location', 'Best');
ax = gca;  % Get handle to current axes.
ax.FontSize = 40;  % Set font size.

hold off;


% % Compute errors
% % Compute NRMSE for x_t_nominal wrt x_t
% residuals_circle = x_t_nominal - x_t;
% NRMSE_circle = sqrt(mean(residuals_circle.^2)) / (max(x_t) - min(x_t));
% disp(['NRMSE for x_t_nominal: ', num2str(NRMSE_circle)])
% 
% % Compute NRMSE for x_t_opt wrt x_t
% residuals_opt = x_t_opt - x_t;
% NRMSE_opt = sqrt(mean(residuals_opt.^2)) / (max(x_t) - min(x_t));
% disp(['NRMSE for x_t_opt: ', num2str(NRMSE_opt)])
% 
% % Compute NRMSE for y_t_nominal wrt y_t
% residuals_circle = y_t_nominal - y_t;
% NRMSE_circle = sqrt(mean(residuals_circle.^2)) / (max(y_t) - min(y_t));
% disp(['NRMSE for y_t_nominal: ', num2str(NRMSE_circle)])
% 
% % Compute NRMSE for y_t_opt wrt y_t
% residuals_opt = y_t_opt - y_t;
% NRMSE_opt = sqrt(mean(residuals_opt.^2)) / (max(y_t) - min(y_t));
% disp(['NRMSE for y_t_opt: ', num2str(NRMSE_opt)])
% 
% % Compute NRMSE for alpha_nominal wrt alpha
% residuals_circle = alpha_nominal - alpha;
% NRMSE_circle = sqrt(mean(residuals_circle.^2)) / (max(alpha) - min(alpha));
% disp(['NRMSE for alpha_nominal: ', num2str(NRMSE_circle)])
% 
% % Compute NRMSE for alpha_opt wrt alpha
% residuals_opt = alpha_opt - alpha;
% NRMSE_opt = sqrt(mean(residuals_opt.^2)) / (max(alpha) - min(alpha));
% disp(['NRMSE for alpha_opt: ', num2str(NRMSE_opt)])




% % Calculate strains
% epsilon_arc_length = (arc_length_f_resampled - L) / L;
% epsilon_circle_model = (circle_model_elongation - L) / L;
% epsilon_opt_model = (opt_model_elongation - L) / L;
% 
% % Elongation
% figure('Name','Model Validation: Strain');
% hold on
% grid off
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





