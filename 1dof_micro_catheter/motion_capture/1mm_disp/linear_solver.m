clear all;
close all
clc

% Find the best shape and the model is finished
silicon_shape = ['circle', 'ellipse', 'shifted ellipse']; % 'semi circle',

syms P 'real';
P = P*1e6; % [Pa] -> [MPa]

%% Data 
% V: Verified data
% X: To be checked
% Initial length
L = 5*1e-3; % [m] V

% Initial Curvature
k_0 = 0; % [1/m] V

% Outer radii
Rc = 0.4*1e-3; % [m] V
Rh = 0.15*1e-3; % [m] V

% Inner radii
Rci = 0.3*1e-3; % [m] V
Rhi = 0.075*1e-3; % [m] V

% Area
% External coil
Ac = pi*(Rc^2 - Rci^2); 

% Hollow channel
Ah = pi*(Rh^2 - Rhi^2); 


if ismember('circle',silicon_shape)
    % Area 
    % Circle with same dimensions of hollow channel section
    Rs = 0.15*1e-3; % [m] V constrained if shape is fixed
    Rsi = 0.085*1e-3; % [m] X

    As = pi*(Rs^2 - Rsi^2); % (SHAPE DEPENDENT)
    Ap = pi*(Rsi^2);  % (SHAPE DEPENDENT)

    % % Total areas
    % Ash = As + Ah;
    % Acsh = Ac + Ash;
    
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
    Eh = spring_h*L/Ah; % [Pa] 
    
    % Moment of inertia
    % Height of the centroid of every section
    yc = Rc; 
    yh = Rh + Rc - Rci;
    ys = Rc + Rs; % (SHAPE DEPENDENT)
    
    % % Compute equivalent area (silicon as reference material)
    % Asn = As*Es/Es;
    % Ahn = Ah*Eh/Es;
    % Acn = Ac*Ec/Es;
    % Acshn = Asn + Ahn + Acn;
    
    % Neutral axis
    % y_bar_area = (ys*As + yh*Ahn + yc*Acn)/(As + Ahn + Acn);
    y_bar_parallel = (ys*Es*As + yh*Eh*Ah + yc*Ec*Ac)/(Es*As + Eh*Ah + Ec*Ac);
    y_bar = y_bar_parallel;
    %y_bar = Rc;
    
    % Distances from centroid to neutral axis (take absolute value)
    dc = abs(yc - y_bar);
    ds = abs(ys - y_bar);
    dh = abs(yh - y_bar);
    
    % Inertia moments
    Ic = pi*(Rc^4 - Rci^4)/4 + Ac*dc^2; 
    Ih = pi*(Rh^4 - Rhi^4)/4 + Ah*dh^2 ;
    Is = pi*(Rs^4 - Rsi^4)/4 + As*ds^2 ; % (SHAPE DEPENDENT)

    % Inputs
    Fp = P*Ap; % [N]
    
    %% Simplest case (consider the structure as a whole)
    %Input moment from pressure
    hp = Rc + Rs;% + 0.0003*1e-3; % [m] (center of the silicon tube)
    e = hp - y_bar; % [m] (center of the silicon tube - neutral axis)
    M = Fp * e; % [Nm]
    
    disp("")
    disp("Circle")
    fprintf('Area = %s\n', num2str(double(vpa(As)), '%.0e'));
    fprintf('Inertia = %s\n', num2str(double(vpa(Is)), '%.0e'));
    fprintf('Pressure height = %s\n', num2str(double(vpa(hp)), '%.0e'));
    fprintf('Neutral axis = %s\n', num2str(double(vpa(y_bar)), '%.0e'));      
    disp("Axial elongation (target coef: 0.111067)")
    % epsilon_area = Fp / (Es*Acshn); % with area transformation
    % epsilon_area = simplify(epsilon_area);
    
    epsilon_parallel = Fp / (Es*As + Ec*Ac + Eh*Ah); % parallel springs    
    epsilon_parallel = simplify(epsilon_parallel);
    %fprintf('Transformed cross-section method:\nepsilon = %s\n',char(vpa(epsilon_area))) % 0.111067
    fprintf('epsilon = %s\n',char(vpa(epsilon_parallel))) % 0.111067
    disp("Curvature (target coef: 0.038811)")
    k_parallel = M / (Es*Is + Ec*Ic + Eh*Ih)*1e-3;
    k_parallel = simplify(k_parallel);
    fprintf('k = %s\n',char(vpa(k_parallel))) % 0.038811

end 
if ismember('semi circle',silicon_shape)
    % Area
    Rs = 0.3*1e-3; % [m] X
    Rsi = 0.225*1e-3; % [m] X
    As = pi*( (Rs)^2 - (Rsi)^2 ) / 2 +  2*Rs*(Rs-Rsi);     
    Ap = pi*(Rsi^2)/2; 
    
    % Young's modulus
    Es = 1.648e6; % 
    spring_c = 0.035; % [N/m] 
    Ec = spring_c*L/Ac; % [Pa]
    spring_h = 0.035; % [N/m] 
    Eh = spring_h*L/Ah; % [Pa] 
    
    % Moment of inertia
    % Height of the centroid of every section
    yc = Rc;
    yh = Rh + Rc - Rci;
    ys = Rc + 4*Rsi/(3*pi) + (Rs - Rsi);
    
    % Neutral axis
    y_bar = (ys*Es*As + yh*Eh*Ah + yc*Ec*Ac)/(Es*As + Eh*Ah + Ec*Ac);
    
    % Distances from centroid to neutral axis (take absolute value)
    dc = abs(yc - y_bar);
    ds = abs(ys - y_bar);
    dh = abs(yh - y_bar);
    
    % Inertia moments
    Ic = pi*(Rc^4 - Rci^4)/4 + Ac*dc^2; 
    Ih = pi*(Rh^4 - Rhi^4)/4 + Ah*dh^2 ;        
    Is = pi*( (Rs)^4 - (Rsi)^4 )/8  + As*ds^2 ; 

    % Inputs
    Fp = P*Ap; % [N]
    
    %% Simplest case (consider the structure as a whole)
    %Input moment from pressure
    hp = Rc + 4*Rsi/(3*pi) + Rs - Rsi; % [m] (center of the silicon tube)
    e = hp - y_bar; % [m] (center of the silicon tube - neutral axis)
    M = Fp * e; % [Nm]
    
    disp(" ")
    disp("Semi circle")
    fprintf('Area = %s\n', num2str(double(vpa(As)), '%.0e'));
    fprintf('Inertia = %s\n', num2str(double(vpa(Is)), '%.0e'));
    fprintf('Pressure height = %s\n', num2str(double(vpa(hp)), '%.0e'));
    fprintf('Neutral axis = %s\n', num2str(double(vpa(y_bar)), '%.0e'));  
    disp("Axial elongation (target coef: 0.111067)")
    epsilon_parallel = Fp / (Es*As + Ec*Ac + Eh*Ah); % parallel springs
    epsilon_parallel = simplify(epsilon_parallel);
    fprintf('epsilon = %s\n',char(vpa(epsilon_parallel))) % 0.111067
    disp("Curvature (target coef: 0.038811)")
    k_parallel = M / (Es*Is + Ec*Ic + Eh*Ih)*1e-3;
    k_parallel = simplify(k_parallel);
    fprintf('k = %s\n',char(vpa(k_parallel))) % 0.038811
end
if ismember('ellipse',silicon_shape)
    % Find the optimal shape by solving
    % min_(x in X) a(k_parallel(x) - k_data)^2 + b(epsilon_parallel(x) - epsilon_data)^2
    % with x = [a, ai, bi] and X = [a_min, a_max]x[ai_min, ai_max]x[bi_min, bi_max]
    a = 0.1*1e-3; % [m] X
    b = 0.15*1e-3; % [m] V constrained if shape is fixed
    ai = 0.051*1e-3; % [m] X
    bi = 0.073*1e-3; % [m] X
    As = pi*(a*b - ai*bi);
    Ap = pi*(ai*bi);
    
    % Young's modulus
    Es = 1.648e6; % 
    spring_c = 0.035; % [N/m] 
    Ec = spring_c*L/Ac; % [Pa]
    spring_h = 0.035; % [N/m] 
    Eh = spring_h*L/Ah; % [Pa] 
    
    % Moment of inertia
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

    % Inputs
    Fp = P*Ap; % [N]
    
    %% Simplest case (consider the structure as a whole)
    %Input moment from pressure
    hp = Rc + b;% + 0.0003*1e-3; % [m] (center of the silicon tube)
    e = hp - y_bar; % [m] (center of the silicon tube - neutral axis)
    M = Fp * e; % [Nm]
    
    disp(" ")
    disp("Ellipse")
    fprintf('Area = %s\n', num2str(double(vpa(As)), '%.0e'));
    fprintf('Inertia = %s\n', num2str(double(vpa(Is)), '%.0e'));
    fprintf('Pressure height = %s\n', num2str(double(vpa(hp)), '%.0e'));
    fprintf('Neutral axis = %s\n', num2str(double(vpa(y_bar)), '%.0e'));  
    disp("Axial elongation (target coef: 0.111067)")
    epsilon_parallel = Fp / (Es*As + Ec*Ac + Eh*Ah); % parallel springs
    epsilon_parallel = simplify(epsilon_parallel);
    fprintf('epsilon = %s\n',char(vpa(epsilon_parallel))) % 0.111067
    disp("Curvature (target coef: 0.038811)")
    k_parallel = M / (Es*Is + Ec*Ic + Eh*Ih)*1e-3;
    k_parallel = simplify(k_parallel);
    fprintf('k = %s\n',char(vpa(k_parallel))) % 0.038811

end

if ismember('shifted ellipse', silicon_shape)

    % Find the optimal shape by solving
    % min_(x in X) a(k_parallel(x) - k_data)^2 + b(epsilon_parallel(x) - epsilon_data)^2
    % with x = [a, ai, bi, h] and X = [a_min, a_max]x[ai_min, ai_max]x[bi_min, bi_max]x[h_min, h_max]
    a = 0.1*1e-3; % [m] X
    b = 0.15*1e-3; % [m] V constrained if shape is fixed
    ai = 0.051*1e-3; % [m] X
    bi = 0.073*1e-3; % [m] X
    h = Rc + b + 1e-7;%0.0003*1e-3; % [m] X

    As_ext = pi*(a*b);
    As_int = pi*(ai*bi);
    As = As_ext - As_int;
    Ap = pi*(ai*bi);
    
    % Young's modulus
    Es = 1.648e6; % 
    spring_c = 0.035; % [N/m] 
    Ec = spring_c*L/Ac; % [Pa]
    spring_h = 0.035; % [N/m] 
    Eh = spring_h*L/Ah; % [Pa] 
    
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
    e = h - y_bar; % [m] (center of the silicon tube - neutral axis)
    M = Fp * e; % [Nm]
    
    disp(" ")
    disp("Shifted Ellipse")
    fprintf('Area = %s\n', num2str(double(vpa(As)), '%.0e'));
    fprintf('Inertia = %s\n', num2str(double(vpa(Is)), '%.0e'));
    fprintf('Pressure height = %s\n', num2str(double(vpa(hp)), '%.0e'));
    fprintf('Neutral axis = %s\n', num2str(double(vpa(y_bar)), '%.0e'));  
    disp("Axial elongation (target coef: 0.111067)")
    epsilon_parallel = Fp / (Es*As + Ec*Ac + Eh*Ah); % parallel springs
    epsilon_parallel = simplify(epsilon_parallel);
    fprintf('epsilon = %s\n',char(vpa(epsilon_parallel))) % 0.111067
    disp("Curvature (target coef: 0.038811)")
    k_parallel = M / (Es*Is + Ec*Ic + Eh*Ih)*1e-3;
    k_parallel = simplify(k_parallel);
    fprintf('k = %s\n',char(vpa(k_parallel))) % 0.038811
end


%% Solve symbolic system of equations
% Consider tube and hollow channel as parallel beams (same formulation as
%parallel springs) MODIFY WITH NEW FORMULATION
% M = [- 1/(Es*As + Eh*Ah) - 1/(Ec*Ac), - 1/(Es*As + Eh*Ah) - 1/(Ec*Ac);
%     -(2*Rh+2*Rs-hc)/(Es*Is + Eh*Ih) + 1/(Ec*Ic), + hc*(1/(Es*Is + Eh*Ih) - 1/(Ec*Ic));];
% 
% Y = [-Fp*(Es*As + Eh*Ah); -Fp*(2*Rh+Rs-hc)/(Es*Is + Eh*Ih);];
% 
% disp("##### Two systems problem results #####")
% N = [M, Y];
% disp('rank(M):')
% disp(rank(M))
% disp('rank(N):')
% disp(rank(N))
% 
% solution = simplify(subs(linsolve(M, Y)));
% disp(solution)
% 
% Ts = solution(1);
% Th = solution(2);
% 
% % Results
% epsilon = (Ts+Th)/(Ec*Ac);
% epsilon = simplify(subs(epsilon));
% vpa(epsilon)
% 
% kc = Rc*(Ts-Th)/(Ec*Ac);
% kc = simplify(subs(kc));
% vpa(kc)

% epsilon = simplify(subs(epsilon));
% L_p = (epsilon + L)*1e3; %[m] -> [mm]
% disp("L_p = ")
% vpa(L_p) %0.111067
% 
% kc = simplify(subs(kc));
% k_p = (kc + k_0)*1e-3; %[1/m] -> [1/mm]
% disp("k_p = ")
% vpa(k_p) % 0.038811

%% Considerd three bodies separately
% % Define M and Y matrices
% M = [1/(Es*As), 1/(Ec*Ac)+1/(Es*As), 1/(Ec*Ac);
%     -1/(Es*As)-1/(Eh*Ah), -1/(Es*As), 1/(Eh*Ah);
%     -1/(Eh*Ah), 1/(Ec*Ac), 1/(Ec*Ac)+1/(Eh*Ah);
%     0, Rc/(Ec*Ic)+2*Rs/(Es*Is), -Rc/(Ec*Ic);
%     0, -2*Rs/(Es*Is), -2*Rh/(Eh*Ih);
%     0, Rc/(Ec*Ic), -Rc/(Ec*Ic)-2*Rh/(Eh*Ih);
% 
% ];
% 
% X = [xc; xs; xh];
% Y = [Fp/(Es*As), -Fp/(Es*As), 0, Rs*Fp/(Es*Is), -Rs*Fp/(Es*Is), 0]';
% 
% N = [M, Y];
% disp(M)
% disp(N)
% 
% % Rouchè-Capelli:
% % Unique solution if and only if rank(M) = rank(N) 
% % Infinite solutions if rank(M) < rank(N)
% % No solutions if rank(M) > rank(N)
% 
% disp('rank(M):')
% disp(rank(M))
% disp('rank(N):')
% disp(rank(N))
% 
% 
% % Solve the system of linear equations for X
% solution = simplify(subs(linsolve(M, Y)));
% disp(solution)
% 
% % Elongation only
% M = M((1:3),:);
% Y = Y(1:3);
% N = [M, Y];
% disp(M)
% disp(N)
% 

% 
% 
% % Solve the system of linear equations for X
% solution = simplify(subs(linsolve(M, Y)));
% disp(solution)