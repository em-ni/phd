clear;
clc
% Load the Symbolic Math Toolbox
syms xc xs xh Fp 'real';
syms Rc Rs Rh Ec Es Eh Ic Is Ih Ac As Ah 'real'

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


% Consider tube and hollow channel as parallel beams (same formulation as
% parallel springs)
M = [1/(Ec*Ac) + Es*As + Eh*Ah, 1/(Ec*Ac) + Es*As + Eh*Ah;
    Rc/(Ec*Ic) + 2*Rs*(Es*Is+Eh*Ih), -Rc/(Ec*Ic) - 2*Rh*(Es*Is+Eh*Ih)];

Y = [Fp*(Es*As+Eh*Ah); Fp*Rs*(Es*Is+Eh*Ih)];

N = [M, Y];
disp('rank(M):')
disp(rank(M))
disp('rank(N):')
disp(rank(N))

solution = simplify(subs(linsolve(M, Y)));
disp(solution)

Ts = solution(1);
Th = solution(2);

% Results
epsilon = (Ts+Th)/(Ec*Ac);
epsilon = simplify(subs(epsilon));
vpa(epsilon)

kc = Rc*(Ts-Th)/(Ec*Ac);
kc = simplify(subs(kc));
vpa(kc)

% Subsitute data 
syms P 'real'
P = P*1e6; % [MPa]

% Initial length
L = 0.005; % [m]

% Initial curvature
k_0 = 0; % [1/m]

% Outer radii
Rc = 0.0004; % [m]
Rs = 0.00015; % [m]
Rh = 0.00015; % [m]

% Inner radii
Rci = 0.0003; % [m]
Rsi = 0.00001; % [m]
Rhi = 0.00001; % [m]

Ec = 1.2*1e9; % Find
Es = 0.05*1e9; % 0.001 - 0.05 GPa
Eh = 1.2*1e9; % Find

Ac = pi*Rc^2;
As = pi*Rs^2; 
Ah = pi*Rh^2; 

Fp = P*As;

Ic = pi*(0.0008^4 - 0.0006^4)/64; % pi*(do^4 - di^4)/64 ????
%Is = 1e1; %*pi*(0.0003^4)/64;
Ih = 1e1; %pi*(0.0003^4 - 0.0001^4)/64;

epsilon = simplify(subs(epsilon));
L_p = (epsilon + L)*1e3; %[mm]
vpa(L_p) %0.111067

kc = simplify(subs(kc));
k_p = (kc + k_0)*1e-3; %[1/mm]
vpa(k_p) % 0.038811

P = 1;
k_p = simplify(subs(k_p));
fplot(k_p)

