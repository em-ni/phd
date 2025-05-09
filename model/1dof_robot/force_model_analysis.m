close all;
clear all;
clc;

addpath(genpath('./functions'));

L = 8e-3;

% Take maximum curvature given q = [0, pi]
max_curvature = 0.5*pi/L;
pressure = linspace(1e4, 1e6, 10); % Span pressure from 0 to 1 MPa
curvature_range = linspace(50, max_curvature, 10); % Range of curvature values

% Preallocate storage for force data
fx_data = zeros(length(pressure), length(curvature_range));
fy_data = zeros(length(pressure), length(curvature_range));
force_pred_abs_data = zeros(length(pressure), length(curvature_range));
f_vertical = zeros(length(pressure), length(curvature_range));
f_horizontal = zeros(length(pressure), length(curvature_range));

% Initialize loading bar
h = waitbar(0, 'Calculating forces...');

% Loop over curvature values
for j = 1:length(curvature_range)
    k = curvature_range(j);
    % Loop over pressure values
    for i = 1:length(pressure)
        [fx, fy] = force_model(L, k, pressure(i));
        fx_data(i, j) = fx; % Store Fx for this curvature and pressure
        fy_data(i, j) = fy; % Store Fy for this curvature and pressure
        force_pred_abs_data(i, j) = sqrt(fx^2 + fy^2); % Store absolute force

        f_vertical(i, j) = fx * cos(k * L) + fy * sin(k * L);
        f_horizontal(i, j) = fx * sin(k * L) - fy * cos(k * L);
        
        % Update loading bar
        progress = ((j-1) * length(pressure) + i) / (length(curvature_range) * length(pressure));
        waitbar(progress, h);
    end
end

% Close loading bar
close(h);

% Plot force components for different curvatures - Fx
% Prepare data and color mapping
colors = jet(length(curvature_range)); % Get a colormap to differentiate curvatures

%% Create a figure for subplots
figure;

pressure = pressure / 1e5; % Convert pressure to MPa for display
fx_data = fx_data * 1e3;   % Convert Fx to milliNewtons for display
fy_data = fy_data * 1e3;   % Convert Fy to milliNewtons for display

% Subplot for Fx
subplot(1,2,1); % Positioning this plot in the first half of the figure
hold on;
for j = 1:length(curvature_range)
    p = plot(pressure, fx_data(:, j), 'Color', colors(j,:), 'LineWidth', 1.5);
    % Label only the first and last curvature values
    if j == 1
        midPoint = floor(length(pressure)/3*2); % Find the middle index of the pressure array
        text(pressure(midPoint), fx_data(midPoint, j) + 0.5, ['k=' num2str(floor(curvature_range(j)))], ...
            'FontSize', 30, 'Color', colors(j,:), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    end
    if j == length(curvature_range)
        midPoint = floor(length(pressure)/3*2); % Find the middle index of the pressure array
        text(pressure(midPoint) + 1, fx_data(midPoint, j) - 0.5, ['k=' num2str(floor(curvature_range(j)))], ...
            'FontSize', 30, 'Color', colors(j,:), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    end
end
xlabel('Pressure [MPa]', 'FontSize', 40);
ylabel('F$^b_x$ [mN]', 'Interpreter', 'latex', 'FontSize', 40);
set(gca, 'FontSize', 40);
hold off;

% Subplot for Fy
subplot(1,2,2); % Positioning this plot in the second half of the figure
hold on;
for j = 1:length(curvature_range)
    p = plot(pressure, fy_data(:, j), 'Color', colors(j,:), 'LineWidth', 1.5);
    % Label only the first and last curvature values
    if j == 1
        midPoint = floor(length(pressure)/3*2); % Find the middle index of the pressure array
        text(pressure(midPoint), fy_data(midPoint, j) - 3, ['k=' num2str(floor(curvature_range(j)))], ...
            'FontSize', 30, 'Color', colors(j,:), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    end
    if j == length(curvature_range)
        midPoint = floor(length(pressure)/3*2); % Find the middle index of the pressure array
        text(pressure(midPoint) + 0.75, fy_data(midPoint, j), ['k=' num2str(floor(curvature_range(j)))], ...
            'FontSize', 30, 'Color', colors(j,:), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    end
end
xlabel('Pressure [MPa]', 'FontSize', 40);
ylabel('F$^b_y$ [mN]', 'Interpreter', 'latex', 'FontSize', 40);
set(gca, 'FontSize', 40);
hold off;



% % Plot f vertical for different curvatures (in case it is in the ee reference frame)
% figure
% hold on
% colors = jet(length(curvature_range)); % Get a colormap to differentiate curvatures
% for j = 1:length(curvature_range)
%     plot(pressure, f_vertical(:,j), 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', ['F vertical, k=' num2str(floor(curvature_range(j)))]);
% end
% xlabel('Pressure [Pa]');
% ylabel('Force [N]');
% legend('Location', 'best');
% title('F$^{ee}_y$ vs pressure for different curvatures', 'Interpreter', 'latex');
% 
% % Plot f horizontal for different curvatures (in case it is in the ee reference frame)
% figure
% hold on
% colors = jet(length(curvature_range)); % Get a colormap to differentiate curvatures
% for j = 1:length(curvature_range)
%     plot(pressure, f_horizontal(:,j), 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', ['F horizontal, k=' num2str(floor(curvature_range(j)))]);
% end
% xlabel('Pressure [Pa]');
% ylabel('Force [N]');
% legend('Location', 'best');
% title('F$^{ee}_x$ vs pressure for different curvatures', 'Interpreter', 'latex');

% % Plot absolute force for different curvatures
% figure
% hold on
% for j = 1:length(curvature_range)
%     plot(pressure, force_pred_abs_data(:, j), 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', ['|F|, k=' num2str(floor(curvature_range(j)))]);
% end
% xlabel('Pressure [Pa]');
% ylabel('Force [N]');
% legend('Location', 'best');
% title('$$|F|$$ vs pressure for different curvatures', 'Interpreter', 'latex');


% %% Perform thje force model analysis for epsilon force model
% L0 = 8e-3;
% pressure = linspace(1e4, 7e5, 10); % Span pressure from 0 to 1 MPa
% L_range = linspace(L0, 10e-3, 10); % Range of length values
% 
% % Preallocate storage for force data
% f_data = zeros(length(pressure), length(L_range));
% f_y_epsilon = zeros(length(pressure), length(L_range));
% 
% % Take a curavture value 
% k = 0.4*pi/L0;
% 
% % Initialize loading bar
% h = waitbar(0, 'Calculating forces...');
% % Loop over length values
% for j = 1:length(L_range)
%     L = L_range(j);
%     % Loop over pressure values
%     for i = 1:length(pressure)
%         f = force_model_epsilon(L, L0, pressure(i));
%         f_data(i, j) = f; % Store F for this length and pressure
%         f_y_epsilon(i, j) = f * sin(k*L);
% 
%         % Update loading bar
%         progress = ((j-1) * length(pressure) + i) / (length(L_range) * length(pressure));
%         waitbar(progress, h);
%     end
% end
% 
% % Close loading bar
% close(h);
% 
% % Plot force components for different lengths
% figure
% hold on
% colors = jet(length(L_range)); % Get a colormap to differentiate lengths
% for j = 1:length(L_range)
%     plot(pressure, f_data(:, j), 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', ['F, L=' num2str(L_range(j)*1e3) 'mm']);
% end
% xlabel('Pressure [Pa]');
% ylabel('Force [N]');
% legend('Location', 'best');
% title('F (elongation) vs pressure for different lengths', 'Interpreter', 'latex');
% 
% % Plot f_y for different lengths
% figure
% hold on
% colors = jet(length(L_range)); % Get a colormap to differentiate lengths
% for j = 1:length(L_range)
%     plot(pressure, f_y_epsilon(:, j), 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', ['F_y, L=' num2str(L_range(j)*1e3) 'mm']);
% end
% xlabel('Pressure [Pa]');
% ylabel('Force [N]');
% legend('Location', 'best');
% title('F$^{b}_y$ (elongation) vs pressure for different lengths', 'Interpreter', 'latex');



