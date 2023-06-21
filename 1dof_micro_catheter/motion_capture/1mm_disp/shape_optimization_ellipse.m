%Data 
clear all;
close all;
clc;

% Initial length
L = 5*1e-3; 

% Outer radii
Rc = 0.4*1e-3;  
Rh = 0.15*1e-3;  

% Inner radii
Rci = 0.3*1e-3;  
Rhi = 0.075*1e-3;  

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
k_parallel = @(a, ai, bi) calculate_k(a, ai, bi, Ac, Ah, Es, Ec, Eh, Rc, Rh, Rci, Rhi, P);
epsilon_parallel = @(a, ai, bi) calculate_epsilon(a, ai, bi, Ac, Ah, Es, Ec, Eh, P);

% Objective function
k_data = 0.038811; 
epsilon_data = 0.111067; 

% Define the objective function
% x = [a, ai, bi]
w_k = 1000;
w_epsilon = 1;
fun = @(x) (  w_k*(k_parallel(x(1), x(2), x(3)) - k_data)^2 + w_epsilon*(epsilon_parallel(x(1), x(2), x(3)) - epsilon_data)^2    );

% Set the constraints
% x = [a, ai, bi]
lb = 1e-3*[0.15, 0.05, 0.05];
ub = 1e-3*[0.2, 0.19, 0.14];

% Define the initial conditions
X0 = 1e-3*[1e3*lb; 1e3*ub; lb + (ub-lb).*rand(1,3); lb + (ub-lb).*rand(1,3); lb + (ub-lb).*rand(1,3); lb + (ub-lb).*rand(1,3)];

% Define array to store results
results = [];

% Loop over initial conditions
for i = 1:size(X0,1)
    % Choose a starting point
    x0 = X0(i,:);

    % Call the constrained optimization function
    x = fmincon(fun, x0, [], [], [], [], lb, ub);

    % Display the result
    disp('Optimized parameters:');
    disp(x);
    
    % Calculate and display k and epsilon
    disp('Calculated k:');
    disp(k_parallel(x(1), x(2), x(3)));
    disp('Calculated epsilon:');
    disp(epsilon_parallel(x(1), x(2), x(3)));

    % Calculate k and epsilon
    k_val = k_parallel(x(1), x(2), x(3));
    epsilon_val = epsilon_parallel(x(1), x(2), x(3));
    
    % Calculate error
    error = abs(k_val - k_data) + abs(epsilon_val - epsilon_data);
    
    % Store results
    results = [results; x, k_val, epsilon_val, error];

    % Very good results! Starting from the boundary of the search space every solution converges to the same result, I supposed that it's a global minimum

end

% Sort results by error (column 6)
sorted_results = sortrows(results,6);

% Display the ranked results
disp('Ranked results (parameters a, ai, bi, calculated k, calculated epsilon, error):');
% Loop over the sorted results and display each result
for i = 1:size(sorted_results, 1)
    fprintf('a = %.3f*1e-3, ai = %.3f*1e-3, bi = %.3f*1e-3, k = %.3f, epsilon = %.3f, error = %.3e\n', ...
        sorted_results(i, 1)*1e3, sorted_results(i, 2)*1e3, sorted_results(i, 3)*1e3, ...
        sorted_results(i, 4), sorted_results(i, 5), sorted_results(i, 6));
end


% Define function to calculate k
function k = calculate_k(a, ai, bi, Ac, Ah, Es, Ec, Eh, Rc, Rh, Rci, Rhi, P)

    b = 0.15*1e-3;
    As = pi*(a*b - ai*bi);
    Ap = pi*(ai*bi);

    % Height of the centroid of every section
    yc = Rc;
    yh = Rh + Rc - Rci;
    ys = Rc + b;

    % Neutral axis
    y_bar = (ys*Es*As + yh*Eh*Ah + yc*Ec*Ac)/(Es*As + Eh*Ah + Ec*Ac);

    % Distances from centroid to neutral axis (take absolute value)
    dc = abs(yc - y_bar);
    ds = abs(ys - y_bar);
    dh = abs(yh - y_bar);

    % Inertia moments
    Ic = pi*(Rc^4 - Rci^4)/4 + Ac*dc^2; 
    Ih = pi*(Rh^4 - Rhi^4)/4 + Ah*dh^2 ;
    Is = pi*(a*b^3 - ai*bi^3)/4 + As*ds^2;   

    % Input moment from pressure
    hp = Rc + b;
    e = hp - y_bar; 
    Fp = P*Ap; 
    M = Fp * e; 
    
    k = M / (Es*Is + Ec*Ic + Eh*Ih)*1e-3;
end

% Define function to calculate epsilon
function epsilon = calculate_epsilon(a, ai, bi, Ac, Ah, Es, Ec, Eh, P)
    b = 0.15*1e-3; % [m] V constrained if shape is fixed
    As = pi*(a*b - ai*bi);
    Ap = pi*(ai*bi);
    
    % Input moment from pressure
    Fp = P*Ap;

    epsilon = Fp / (Es*As + Ec*Ac + Eh*Ah);
end
