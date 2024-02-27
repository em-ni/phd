clear all
close all
clc

% Length
L = 8*1e-3; % [m]

% Get data from large bending experiment
% Pressure [Bar], Force [g], Radius [px], Curvature [1/px], Arc Length [px], x_tip [px], y_tip [px], x_base [px], y_base [px]
data = load("data/force_small_bending/cv_output.csv");
%data = load("data/force_large_bending/cv_output.csv");

% [Bar] to [Pa]
pressure_bar = data(:,1); % [Bar]
pressure_mpa = pressure_bar*0.1; % [MPa]
pressure = pressure_mpa*1e6; % [Pa]

% [g] to [N]
force = data(:,2); % grams % vertical force in the world frame (opencv frame) = fy in world frame = fy in base frame
force_data = force*0.00980665; % [N]

% [px] to [m]
radius = data(:,3);
arc_length = data(:,5);
x_tip = data(:,6);
y_tip = data(:,7);
x_base = data(:,8);
y_base = data(:,9);

% Get conversion rate from px to m
arc_length_rest = min(arc_length);
conv_rate = L/arc_length_rest; % [m/px]

% Convert to m
arc_length = arc_length*conv_rate; % [m]
radius = radius*conv_rate; % [m]
curvature = 1 ./ radius; % [1/m]
x_tip = x_tip*conv_rate; % [m]
y_tip = y_tip*conv_rate; % [m]
x_base = x_base*conv_rate; % [m]
y_base = y_base*conv_rate; % [m]


% % Plot pressure vs curvature
% figure
% hold on
% ylim([0, 250])
% plot(pressure, curvature, 'o', 'DisplayName', 'Curvature')
% xlabel('Pressure [Pa]')
% ylabel('Curvature [1/m]')
% legend
% title('Curvature vs pressure')

% Set all curvatures to the average curvature (in theory it should not change for the CC model)
% Adding a correction factor to the curvature makes the slope ftting better
% So it's worth doing a comparison between ideal curvature and the measured curvature. The latter will be different because the robot will drift from the expected CC model
% Use the tangent line (parallel to the robot body which is horizontal) to compute the ideal curvature
% curvature_cf = 140;
% curvature_cf = 0;
% curvature_avg = mean(curvature);
% for i = 1:length(curvature)
%     curvature(i) = curvature(i) - curvature_cf;
% end


% % Add the averages to the plot
% plot(pressure, curvature, 'o', 'DisplayName', 'Curvature (avg)')
% legend

% % Plot pressure vs arc length
% figure
% hold on
% plot(pressure, arc_length, 'o', 'DisplayName', 'Arc length')
% xlabel('Pressure [Pa]')
% ylabel('Arc length [m]')
% legend
% title('Arc length vs pressure')
% ylim([0, 10*1e-3])

% Set all arc lengths to the average arc length (in theory it should not change for the CC model)
% arc_length_avg = mean(arc_length);
% for i = 1:length(arc_length)
%     arc_length(i) = arc_length_avg;
% end


disp("Data loaded and converted to SI units");
disp(" ");

% Test kinematic model prediction
disp("Kinematic model prediction");
for i = 1:length(pressure)
    % Compute tip prediction (from the base to the tip) and transform to the world frame (opencv frame)
    [x_tip_pred_temp, y_tip_pred_temp] = kin_model(curvature(i), arc_length(i));
    x_tip_pred(i) = x_tip_pred_temp + x_base(i);
    y_tip_pred(i) = y_tip_pred_temp + y_base(i);
    disp('x_tip_pred , y_tip_pred')
    disp([x_tip_pred, y_tip_pred]);

    % Compare with the data (tip position in the world frame)
    x_tip_data = x_tip(i);
    y_tip_data = y_tip(i);
    disp('x_tip_data , y_tip_data')
    disp([x_tip_data, y_tip_data]);
    disp(" ");
end

% % Plot x_tip_pred vs x_tip_data and y_tip_pred vs y_tip_data
% figure

% % Subplot 1: x Tip position vs pressure
% subplot(2, 1, 1)
% hold on
% plot(pressure, x_tip, 'o', 'DisplayName', 'x tip data')
% plot(pressure, x_tip_pred, 'o', 'DisplayName', 'x tip prediction')
% xlabel('Pressure [Pa]')
% ylabel('x Tip position [m]')
% legend
% title('x Tip position vs pressure')
% ylim([0, 10*1e-3])
% 
% % Subplot 2: y Tip position vs pressure
% subplot(2, 1, 2)
% hold on
% plot(pressure, y_tip, 'o', 'DisplayName', 'y tip data')
% plot(pressure, y_tip_pred, 'o', 'DisplayName', 'y tip prediction')
% xlabel('Pressure [Pa]')
% ylabel('y Tip position [m]')
% legend
% title('y Tip position vs pressure')
% ylim([0, 10*1e-3])


% Get force prediction
disp("Force model prediction");
for i = 1:length(pressure)
    % Compute force prediction (from CC dynamic model)
    [fx_pred(i), fy_pred(i)] = force_model(arc_length(i), curvature(i), pressure(i)); % use min(curvature) since it theory the CC is assumed

    % The model predicts the contact force from the environment on the robot
    % Then to get the force on the environment from the robot, we need to negate the force
    % These are the forces that the robot applies on the environment expressed in the ??? frame
    fx_pred(i) = -fx_pred(i);
    fy_pred(i) = -fy_pred(i);

    %disp('fx_pred , fy_pred')
    %disp([fx_pred(i), fy_pred(i)]);
    %disp(" ");
    % Compute the absolute value of the force (useful to compare order of magnitude with the data)
    force_pred_abs(i) = sqrt(fx_pred(i)^2 + fy_pred(i)^2); % [N] 

    % Project on the vertical axis (scale is based on gravity) 
    % This is true if the model prediction is in the end effector frame
    %force_pred(i) = fx_pred(i) * cos(arc_length(i)*curvature(i)) + fy_pred(i) * sin(arc_length(i)*curvature(i));
    % This is true if the model prediction is in the base frame
    force_pred(i) = fy_pred(i);

    % Add a correction factor exponential to pressure/curvature
    %force_pred(i) = force_pred(i) + 0.1*1e-5*exp(2.5*1e-3*pressure(i)/curvature(i));
    %force_pred(i) = force_pred(i) + 1.3*1e-6*pressure(i)/curvature(i);

    % Compute force from the elongation of the robot 
    f_epsilon(i) = force_model_epsilon(arc_length(i), min(arc_length), pressure(i));

    % Take negative for same reason as above
    f_epsilon(i) = -f_epsilon(i);

    % Project on the vertical axis 
    fy_epsilon(i) = f_epsilon(i) * sin(arc_length(i)*curvature(i));

    % Add the force from the elongation of the robot to the force prediction
    alpha = 1e-5;
    beta = 8.2e-3;
    beta = 6.3e-3;
    force_pred(i) = force_pred(i) + alpha*pressure(i)*fy_epsilon(i)/curvature(i) - beta;
    if force_pred(i) < 0
        force_pred(i) = 0;
    end

end

% Plot force components vs pressure
figure
hold on
plot(pressure, fx_pred, 'o', 'DisplayName', 'Fx prediction')
plot(pressure, fy_pred, 'o', 'DisplayName', 'Fy prediction')
xlabel('Pressure [Pa]')
ylabel('Force [N]')
legend
title('Force components vs pressure')

% Plot force pred vs pressure
figure
hold on
plot(pressure, force_pred_abs, 'o', 'DisplayName', 'Force prediction absolute value')
xlabel('Pressure [Pa]')
ylabel('Force [N]')
legend
title('Force prediction vs pressure')

% Plot force pred vs pressure and force data vs pressure
figure
hold on
plot(pressure, force_pred, 'o', 'DisplayName', 'Force prediction vertical component')
plot(pressure, force_data, 'o', 'DisplayName', 'Force data')
xlabel('Pressure [Pa]')
ylabel('Force [N]')
legend
title('Force prediction vs data')

% Plot f epsilon vs pressure
figure
hold on
plot(pressure, f_epsilon, 'o', 'DisplayName', 'F epsilon')
xlabel('Pressure [Pa]')
ylabel('Force [N]')
legend
title('F epsilon vs pressure')

% Plot fy epsilon vs pressure
figure
hold on
plot(pressure, fy_epsilon, 'o', 'DisplayName', 'Fy epsilon')
xlabel('Pressure [Pa]')
ylabel('Force [N]')
legend
title('Fy epsilon vs pressure')







