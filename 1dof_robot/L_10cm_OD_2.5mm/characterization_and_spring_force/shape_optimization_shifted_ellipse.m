% This script finds the optimal parameters for the non-concentric ellipses
% Find the optimal shape by solving
% min_(x in X) a(k_parallel(x) - k_data)^2 + b(epsilon_parallel(x) - epsilon_data)^2
% with x = [a, ai, bi, h] and X = [a_min, a_max]x[ai_min, ai_max]x[bi_min, bi_max]x[h_min, h_max]

%Data 
clear all;
close all;
clc;

% Initial length
L = 10*1e-3; 

% Outer radii
Rc = 1.25*1e-3;  
Rh = 0.75*1e-3;  
Rs_nominal = 0.597*1e-3; % nominal value not stretched
    
% Inner radii
Rci = 1.05*1e-3;  
Rhi = 0.5*1e-3;   
Rsi_nominal = 0.3175*1e-3;  % nominal value not stretched

% Silicon tube (estimated value after stretching)
Rs = (2*Rci - 2*Rh) / 2;
Rsi = 0.15*1e-3;

% Area
% External coil
Ac = pi*(Rc^2 - Rci^2); 

% Hollow channel
Ah = pi*(Rh^2 - Rhi^2); 

% Young's modulus
Es = 1.648e6; 
spring_c = 0.035;
Ec = spring_c*L/Ac;
spring_h = 0.035; 
Eh = spring_h*L/Ah; 

%Inputs
P = 1e6; 

% Function definitions
k_parallel = @(a, ai, bi, h) calculate_k(a, ai, bi, h, Ac, Ah, Es, Ec, Eh, Rc, Rh, Rci, Rhi, P, 0);
epsilon_parallel = @(a, ai, bi, h) calculate_epsilon(a, ai, bi, Ac, Ah, Es, Ec, Eh, Rci, Rh, P);

% Objective function (from data processing)
% Surprisingly, results from the last data processing are quite the same of the ones from the previous data processing
k_data = 0.038811 / 2; 
epsilon_data = 0.111067; 

% Define the objective function
% x = [a, ai, bi, h]
w_k = 100000;
w_epsilon = 1;
fun = @(x) (  w_k*(k_parallel(x(1), x(2), x(3), x(4)) - k_data)^2 + w_epsilon*(epsilon_parallel(x(1), x(2), x(3), x(4)) - epsilon_data)^2    );

% Set the constraints
% x = [a, ai, bi]
min_thick = 0.1*1e-3;
max_thick = Rs_nominal - Rsi_nominal; % Wall thickness cannot be greater than the nominal value
b = (2*Rci - 2*Rh)/2;

a_max = sqrt( 2*Rci*b - b^2 ); % Form geometric considerations, maximum inscribed ellipse in the space between hollow channle and external coil
a_min = b; % a circle in the limit
ai_min = b;
ai_max = a_max - min_thick;
bi_min = b - max_thick;
bi_max = b - min_thick;

h_max = 2*Rc - (Rc - Rci) - min_thick - bi_min; % Outer diameter - coil thicknness - min silicon thickness - min hollow thickness
h_min = (Rc - Rci) + 2*Rh + b; % Coil thickness + hollow channel dimater + b

lb = [a_min, ai_min, bi_min, h_min];
ub = [a_max, ai_max, bi_max, h_max];

% Number of initial conditions
num_initial_conditions = 10000;

% Preallocate X0
X0 = zeros(num_initial_conditions, 4);

% Generate initial conditions
for i = 1:num_initial_conditions
    a = a_min + (a_max - a_min).*rand(1,1);
    ai = ai_min + (a - ai_min).*rand(1,1); % Ensure ai < a by using a as upper bound
    bi = bi_min + (bi_max - bi_min).*rand(1,1);
    h = h_min + (h_max - h_min).*rand(1,1);
    X0(i,:) = [a, ai, bi, h];
end


% Define array to store results
results = [];

% Define the options for fmincon to include the nonlinear constraint
options = optimoptions('fmincon','Algorithm','sqp');

% Loop over initial conditions
for i = 1:size(X0,1)

    % Choose a starting point
    x0 = X0(i,:);

    % Call the constrained optimization function with nonlinear constraints
    [x, fval, exitflag, output] = fmincon(fun, x0, [], [], [], [], lb, ub, @nonlincon, options);

    % Display the result
    % disp('Optimized parameters:');
    disp(x);
    
    % Calculate and display k and epsilon
    % disp('Calculated k:');
    disp(k_parallel(x(1), x(2), x(3), x(4)));
    % disp('Calculated epsilon:');
    disp(epsilon_parallel(x(1), x(2), x(3), x(4)));

    % Calculate k and epsilon
    k_val = k_parallel(x(1), x(2), x(3), x(4));
    epsilon_val = epsilon_parallel(x(1), x(2), x(3), x(4));
    
    % Calculate error
    error = sqrt( mean( (k_val - k_data)^2 + (epsilon_val - epsilon_data)^2) );%uncomment to NRMSE / (max([k_data; epsilon_data]) - min([k_data; epsilon_data]));
    
    % Store results
    results = [results; x, k_val, epsilon_val, error];

    % Very good results! Starting from the boundary of the search space every solution converges to the same result, I supposed that it's a global minimum

end

% Sort results by error (column 6)
sorted_results = sortrows(results,6);

% Display the ranked results
disp('Ranked results (parameters a, ai, bi, calculated k, calculated epsilon, RMSE):');
% Loop over the sorted results and display each result
for i = 1:size(sorted_results, 1)
    fprintf('a = %.3f*1e-3, ai = %.3f*1e-3, bi = %.3f*1e-3, h = %.3f*1e-3, k = %.3f, epsilon = %.3f, error = %.3e\n', ...
        sorted_results(i, 1)*1e3, sorted_results(i, 2)*1e3, sorted_results(i, 3)*1e3, sorted_results(i, 4)*1e3, ...
        sorted_results(i, 5), sorted_results(i, 6) , sorted_results(i, 7));
end

% Save the best result in a .mat file
k_opt = sorted_results(1,5);
epsilon_opt = sorted_results(1,6);
save('data/optimization_results.mat', 'epsilon_opt', 'k_opt');

% Define semiaxes
a = sorted_results(1,1); % semiaxis along x for outer ellipse
b = (2*Rci - 2*Rh)/2; % semiaxis along y for outer ellipse
ai = sorted_results(1,2); % semiaxis along x for inner ellipse
bi = sorted_results(1,3); % semiaxis along y for inner ellipse
h = sorted_results(1,4); % height of hollow channel

% Define angle
theta = linspace(0,2*pi,1000);

% External coil
x_coil_ext = Rc * cos(theta);
y_coil_ext = Rc * sin(theta) + Rc;
x_coil_int = Rci * cos(theta);
y_coil_int = Rci * sin(theta) + Rc;

% Hollow channel
x_hollow_ext = Rh * cos(theta);
y_hollow_ext = Rh * sin(theta) + Rc - Rci + Rh;
x_hollow_int = Rhi * cos(theta);
y_hollow_int = Rhi * sin(theta) + Rc - Rci + Rh - Rhi + Rhi;

% Silicon tube
% Calculate the outer ellipse coordinates
x_outer = a * cos(theta);
y_outer = b * sin(theta) + Rc - Rci + Rh + Rh + b;

% Calculate the inner ellipse coordinates
x_inner = ai * cos(theta);
y_inner = bi * sin(theta) + h;

% Calculate the circular cross section coordinates
x_outer_circ = Rs * cos(theta);
y_outer_circ = Rs * sin(theta) + Rc - Rci + Rh + Rh + Rs;
x_inner_circ = Rsi * cos(theta);
y_inner_circ = Rsi * sin(theta) + Rc - Rci + Rh + Rh + Rs - Rsi + Rsi;

% Create subplot for the circular cross section
subplot('Position', [0.03 0.18 0.5 0.7]);
hold on;
plot(x_outer_circ, y_outer_circ, 'b', 'LineWidth', 2); % Plot outer circle with blue color
plot(x_inner_circ, y_inner_circ, 'b', 'LineWidth', 2); % Plot inner circle with blue color
plot(x_hollow_ext, y_hollow_ext, 'k', 'LineWidth', 3); % Plot hollow channel
plot(x_hollow_int, y_hollow_int, 'k', 'LineWidth', 3); % Plot hollow channel
plot(x_coil_ext, y_coil_ext, 'k', 'LineWidth', 3); % Plot external coil
plot(x_coil_int, y_coil_int, 'k', 'LineWidth', 3); % Plot internal coil
title('Circular Cross Section', 'FontSize', 40);
xlabel('$x$ [$\mu$m]', 'Interpreter', 'latex', 'FontSize', 40); % Keep units as micrometers
ylabel('$y$ [$\mu$m]', 'Interpreter', 'latex', 'FontSize', 40); % Keep units as micrometers
set(gca, 'XTickLabel', sprintfc('%d',get(gca,'XTick')*1e4), 'FontSize', 40); % Correct x-axis units to micrometers
set(gca, 'YTickLabel', sprintfc('%d',get(gca,'YTick')*1e4), 'FontSize', 40); % Correct y-axis units to micrometers
axis equal; % Ensure the x and y axes are equally spaced
%xlim([-5*1e-4 5*1e-4]); % Set the x limits

% Create subplot for the elliptical cross section
subplot('Position', [0.53 0.18 0.5 0.7]);
hold on;
plot(x_outer, y_outer, 'r', 'LineWidth', 2); % Plot outer ellipse with red color
plot(x_inner, y_inner, 'g', 'LineWidth', 2); % Plot inner ellipse with green color
plot(x_hollow_ext, y_hollow_ext, 'k', 'LineWidth', 3); % Plot hollow channel
plot(x_hollow_int, y_hollow_int, 'k', 'LineWidth', 3); % Plot hollow channel
plot(x_coil_ext, y_coil_ext, 'k', 'LineWidth', 3); % Plot external coil
plot(x_coil_int, y_coil_int, 'k', 'LineWidth', 3); % Plot internal coil
title('Ellipse Optimization', 'FontSize', 40);
xlabel('$x$ [$\mu$m]', 'Interpreter', 'latex', 'FontSize', 40); % Keep units as micrometers
ylabel('$y$ [$\mu$m]', 'Interpreter', 'latex', 'FontSize', 40); % Keep units as micrometers
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%d', x*1e4), get(gca, 'XTick'), 'UniformOutput', false), 'FontSize', 40); 
set(gca, 'YTickLabel', arrayfun(@(x) sprintf('%d', x*1e4), get(gca, 'YTick'), 'UniformOutput', false), 'FontSize', 40); 

axis equal; % Ensure the x and y axes are equally spaced
%xlim([-5*1e-4 5*1e-4]); % Set the x limits

set(gcf, 'Position', [0 0 1000 500])  % Resize the figure to accommodate both subplots

% Print final functions
calculate_k(a, ai, bi, h, Ac, Ah, Es, Ec, Eh, Rc, Rh, Rci, Rhi, P, 1);

% Print the best results with final error
disp('Best results:');
fprintf('a = %.3f*1e-3, ai = %.3f*1e-3, bi = %.3f*1e-3, h = %.3f*1e-3, k = %.3f, epsilon = %.3f, error = %.3e\n', ...
    sorted_results(1, 1)*1e3, sorted_results(1, 2)*1e3, sorted_results(1, 3)*1e3, sorted_results(1, 4)*1e3, ...
    sorted_results(1, 5), sorted_results(1, 6), sorted_results(1, 7));

function [c, ceq] = nonlincon(x)
    % Inequality constraints
    c = x(2) - x(1); % This ensures that ai - a < 0, which is equivalent to a > ai
    
    % No equality constraints
    ceq = [];
end


% Define function to calculate k
function k = calculate_k(a, ai, bi, h, Ac, Ah, Es, Ec, Eh, Rc, Rh, Rci, Rhi, P, save)

    b = (2*Rci - 2*Rh)/2;
    As_ext = pi*(a*b);
    As_int = pi*(ai*bi);
    As = As_ext - As_int;
    Ap = pi*(ai*bi);
    
    % Moment of inertia
    % Height of the centroid of every section
    yc = Rc;
    yh = Rh + Rc - Rci;
    ys_ext = Rc + b; % Centroid of external ellipse (real tube)
    ys_int = h; % Centroid of internal ellipse (hole in the tube)
    ys = (ys_ext*As_ext - ys_int*As_int) / As; % Centroid of the whole structure

    % Neutral axis
    y_bar = (ys*Es*As + yh*Eh*Ah + yc*Ec*Ac)/(Es*As + Eh*Ah + Ec*Ac);
    
    % Distances from centroid to neutral axis (take absolute value)
    dc = abs(yc - y_bar);
    dh = abs(yh - y_bar);
    ds_ext = abs(ys_ext - y_bar);
    ds_int = abs(ys_int - y_bar);
    
    % Inertia moments
    Ic = pi*(Rc^4 - Rci^4)/4 + Ac*dc^2; 
    Ih = pi*(Rh^4 - Rhi^4)/4 + Ah*dh^2 ;
    Is_ext = pi*(a*b^3)/4 + As_ext*ds_ext^2;
    Is_int = pi*(ai*bi^3)/4 + As_int*ds_int^2;   
    Is = Is_ext - Is_int;

    % Inputs
    Fp = P*Ap; % [N]
    
    %% Simplest case (consider the structure as a whole)
    %Input moment from pressure
    e = h - y_bar; 
    M = Fp * e; 
    
    k = M / (Es*Is + Ec*Ic + Eh*Ih)*1e-3;

    if save
        fprintf('\nInput torque M(p) = %d p\n', M);
        fprintf('Elastic term K(q) = %d q\n', (Es*Is + Ec*Ic + Eh*Ih)*1e-3);
        fprintf('Curvature k(p) = %d p\n', k);
        disp('Pressure is in MPa, curvature is in mm^-1');
    end
    

end

% Define function to calculate epsilon
function epsilon = calculate_epsilon(a, ai, bi, Ac, Ah, Es, Ec, Eh, Rci, Rh, P)
    b = (2*Rci - 2*Rh)/2;
    As = pi*(a*b - ai*bi);
    Ap = pi*(ai*bi);
    
    % Input moment from pressure
    Fp = P*Ap;

    epsilon = Fp / (Es*As + Ec*Ac + Eh*Ah);
end

