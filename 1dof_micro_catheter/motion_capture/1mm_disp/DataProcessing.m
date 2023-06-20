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
xlabel('Time [s]','fontsize',16)
ylabel('Displacement [mm]','fontsize',16)
legend('Linear Displacement','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Pressure
subplot(2,1,2);
hold on;
grid on;
plot(pressure.Pressure(1,:),pressure.Pressure(2,:),'Linewidth',2);
xlabel('Time [s]','fontsize',16)
ylabel('Pressure [MPa]','fontsize',16)
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
xlabel('Time [s]','fontsize',16)
ylabel('Radius [mm]','fontsize',16)
legend('Radius [mm]','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Curvature
subplot(3,1,2);
hold on;
grid on;
plot(curvature,'Linewidth',2);
xlabel('Time [s]','fontsize',16)
ylabel('Curvature [1/mm]','fontsize',16)
legend('Curvature','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Arc length
subplot(3,1,3);
hold on;
grid on;
plot(arc_length,'Linewidth',2);
xlabel('Time [s]','fontsize',16)
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
xlabel('Time [s]','fontsize',16)
ylabel('Filtered displacement [mm]','fontsize',16)
legend('Linear displacement','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for pressure
subplot(2,1,2);
hold on;
grid on;
plot(pressure.Pressure(1,:),pressure_f,'Linewidth',2);
xlabel('Time [s]','fontsize',16)
ylabel('Filtered pressure [MPa]','fontsize',16)
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
xlabel('Time [s]','fontsize',16)
ylabel('Filtered radius [mm]','fontsize',16)
legend('Radius','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for curvature
subplot(3,1,2);
hold on;
grid on;
plot(curvature_f,'Linewidth',2);
xlabel('Time [s]','fontsize',16)
ylabel('Filtered curvature [1/mm]','fontsize',16)
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
xlabel('Time [s]','fontsize',16)
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
subplot(2,1,1);
hold on;
grid on;
plot(t_new, pressure_f,'Linewidth',2);
xlabel('Time [s]','fontsize',16)
ylabel('Filtered pressure [MPa]','fontsize',16)
legend('Pressure','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for curvature
subplot(2,1,2);
hold on;
grid on;
plot(t_new, curvature_f_resampled,'Linewidth',2);
xlabel('Time [s]','fontsize',16)
ylabel('Filtered curvature [1/mm]','fontsize',16)
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
xlabel('Pressure [MPa]','fontsize',16)
ylabel('Curvature [1/mm]','fontsize',16)
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
slope_epsilon = coefficients(1);
intercept_epsilon = coefficients(2);

% Print the line equation
fprintf('The best fit line is epsilon = %f*P + %f\n', slope_epsilon, intercept_epsilon);

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
xlabel('Time [s]','fontsize',16)
ylabel('Filtered pressure [MPa]','fontsize',16)
legend('Pressure','fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

% Create the second subplot for arc_length
subplot(2,1,2);
hold on;
grid on;
plot(t_new, arc_length_f_resampled,'Linewidth',2);
xlabel('Time [s]','fontsize',16)
ylabel('Filtered arc length [mm]','fontsize',16)
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
xlabel('Pressure [MPa]','fontsize',16)
ylabel('arc length [mm]','fontsize',16)
legend('Pressure vs arc_length', 'Linear approximation', 'fontsize',16)
legend('Location', 'Best');
%set(gca,'FontSize',20);
hold off

%% Modelling
silicon_shape = 'cylinder'; % 'half cylinder' cylinder fits better

% Define symbolic variables
% Unknowns vincular reactions and inputs
syms xc xs xh Fp P 'real';

% Geometry and material properties
syms Rc Rs Rh Ec Es Eh Is Ih Ic Ac As Ah  'real'
syms hc 'real'

%% Substitute Data 
% V: Verified data
% X: To be checked
% Initial length
L = 5*1e-3; % [m] V

% Initial curvature
k_0 = 0; % [1/m] V

% Outer radii
Rc = 0.4*1e-3; % [m] V
Rs = 0.15*1e-3; % [m] X
Rh = 0.15*1e-3; % [m] V

% Inner radii
Rci = 0.3*1e-3; % [m] V
Rsi = 0.075*1e-3; % [m] X
Rhi = 0.075*1e-3; % [m] V

% Area
% External coil
Ac = pi*(Rc^2 - Rci^2); 

% Hollow channel
Ah = pi*(Rh^2 - Rhi^2); 

% Silicon tube
if strcmp(silicon_shape,'cylinder')
    % as hollow cylinder 0.3mm OD 0.15mm ID
    As = pi*(Rs^2 - Rsi^2); 
    Ap = pi*(Rsi^2);  
elseif strcmp(silicon_shape,'half cylinder')
    % as half-hollow cylinder 0.6 mm OD 0.3 mm ID
    As = pi*( (4*Rs)^2 - (4*Rsi)^2 )/2;     
    Ap = pi*(4*Rsi^2)/2; 
end

Ash = As + Ah;
Acsh = Ac + Ash;

% Young's modulus
Es = 1.648e6; % V
% 0.001	0.05 GPa (https://www.azom.com/properties.aspx?ArticleID=920)
% 1.648 MPa; % Cong paper
% 0.387 MPa (A Structural Optimisation Method for a Soft Pneumatic Actuator)  
% 2.69 3.57 3.84 4.51 4.27 MPa (https://www.researchgate.net/publication/314012355_Preparation_and_characterization_of_silicone_rubber_with_high_modulus_via_tension_spring-type_crosslinking)
% Young's modulus from spring constant
% k = F/dL ;  E*A/L = F/dL  -> E = k*L/A
spring_c = 0.035; % [N/m] % Cong paper V
Ec = spring_c*L/Ac; % [Pa]
spring_h = 0.035; % [N/m] 
Eh = spring_h*L/As; % [Pa] 

% Moment of inertia
% Height of the centroid of every section
yc = Rc;
ys = Rc + Rs;
yh = Rh + Rc - Rci;

% Compute equivalent area (silicon as reference material)
Asn = As*Es/Es;
Ahn = Ah*Eh/Es;
Acn = Ac*Ec/Es;
Acshn = Asn + Ahn + Acn;

% Neutral axis
y_bar_area = (ys*As + yh*Ahn + yc*Acn)/(As + Ahn + Acn);
y_bar_parallel = (ys*Es*As + yh*Eh*Ah + yc*Ec*Ac)/(Es*As + Eh*Ah + Ec*Ac);
y_bar = y_bar_parallel;

% Distances from centroid to neutral axis (take absolute value)
dc = abs(yc - y_bar);
ds = abs(ys - y_bar);
dh = abs(yh - y_bar);

% Inertia moments
Ic = pi*(Rc^4 - Rci^4)/4 + Ac*dc^2; 
Ih = pi*(Rh^4 - Rhi^4)/4 + Ah*dh^2 ;

if strcmp(silicon_shape,'cylinder')
    % as hollow cylinder 0.3mm OD 0.15mm ID
    Is = 0.6*pi*(Rs^4 - Rsi^4)/4 + As*ds^2 ; % cylinder
elseif strcmp(silicon_shape,'half cylinder')

    % as half-hollow cylinder 0.6 mm OD 0.3 mm ID
    Is = pi*( (4*Rs)^4 - (4*Rsi)^4 )/8  + As*ds^2 ; % half-hollow cylinder
end 

% Inputs
P = P*1e6; % [Pa] -> [MPa]
Fp = P*Ap; % [N]


%% Simplest case (consider the structure as a whole)
%Input moment from pressure
hp = Rc + Rs; % [m] (center of the silicon tube)
e = hp - y_bar; % [m] (center of the silicon tube - neutral axis)
M = Fp * e; % [Nm]

disp("##### One system problem results #####")
disp("AXIAL ELONGATION (target coef: 0.111067)")
epsilon_area = Fp / (Es*Acshn); % with area transformation
epsilon_parallel = Fp / (Es*As + Ec*Ac + Eh*Ah); % parallel springs
epsilon_area = simplify(epsilon_area);
epsilon_parallel = simplify(epsilon_parallel);
%fprintf('Transformed cross-section method:\nepsilon = %s\n',char(vpa(epsilon_area))) % 0.111067
fprintf('\nParallel spring method: \nepsilon = %s\n',char(vpa(epsilon_parallel))) % 0.111067

disp(" ")
disp("CURVATURE (target coef: 0.038811)")
k_parallel = M / (Es*Is + Ec*Ic + Eh*Ih)*1e-3;
k_parallel = simplify(k_parallel);
fprintf('\nParallel spring method: \nk = %s\n',char(vpa(k_parallel))) % 0.038811

% Display results
epsilon_model = double(subs(epsilon_parallel/P))*1e6;
model_elongation = epsilon_model*pressure_f + intercept_epsilon;

k_model = double(subs(k_parallel/P))*1e6;
model_curvature = k_model*pressure_f + intercept_k;

% Elongation
figure(8);
hold on
plot(pressure_f,model_elongation,'DisplayName','Model','LineWidth', 2)
hold off

% Curvature
figure(6);
hold on
plot(pressure_f,model_curvature,'DisplayName','Model','LineWidth', 2)
hold off







