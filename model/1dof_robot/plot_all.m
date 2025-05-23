clear all
close all
clc
addpath(genpath('./functions')); % Ensure your functions are in this path

% Length
L = 8*1e-3; % [m]

% --- Process Small Bending Experiment Data ---
disp("Loading and processing Small Bending data...");
data_small_raw = load("data/L_10cm_OD_1.5mm/characterization_and_scale_force/force_small_bending/cv_output.csv");

% Pressure [Bar] to [Pa]
pressure_bar_small = data_small_raw(:,1); % [Bar]
pressure_mpa_small = pressure_bar_small*0.1; % [MPa]
pressure_small = pressure_mpa_small*1e6; % [Pa]

% Force [g] to [N]
force_g_small = data_small_raw(:,2); % grams
force_data_small = force_g_small*0.00980665; % [N]

% Pixel to meter conversion
arc_length_px_small = data_small_raw(:,5);
arc_length_rest_small = min(arc_length_px_small);
conv_rate_small = L/arc_length_rest_small; % [m/px]

arc_length_small = arc_length_px_small*conv_rate_small; % [m]
radius_px_small = data_small_raw(:,3);
radius_small = radius_px_small*conv_rate_small; % [m]
curvature_small = 1 ./ radius_small; % [1/m] % Handle potential division by zero if radius can be zero
curvature_small(isinf(curvature_small)) = 0; % Or some other appropriate handling for zero radius

% Force model prediction for Small Bending
disp("Calculating force model prediction for Small Bending...");
fx_pred_small = zeros(size(pressure_small));
fy_pred_small = zeros(size(pressure_small));
force_pred_small = zeros(size(pressure_small));
fy_epsilon_small_arr = zeros(size(pressure_small)); % Renamed to avoid conflict

min_arc_length_small = min(arc_length_small);

for i = 1:length(pressure_small)
    [fx_pred_small(i), fy_pred_small(i)] = force_model(arc_length_small(i), curvature_small(i), pressure_small(i));
    
    fx_pred_small(i) = -fx_pred_small(i);
    fy_pred_small(i) = -fy_pred_small(i);
    
    current_force_pred_small = fy_pred_small(i); % Assuming prediction is in base frame

    f_epsilon_small = force_model_epsilon(arc_length_small(i), min_arc_length_small, pressure_small(i));
    f_epsilon_small = -f_epsilon_small;
    fy_epsilon_small = f_epsilon_small * sin(arc_length_small(i)*curvature_small(i));
    fy_epsilon_small_arr(i) = fy_epsilon_small; % Store for clarity or debugging

    alpha_small_const = 1.4e-5; % Renamed to avoid conflict later
    beta_small_const = 9.3e-3;  % Renamed to avoid conflict later
    force_pred_small(i) = current_force_pred_small + alpha_small_const*pressure_small(i)*fy_epsilon_small/curvature_small(i) - beta_small_const;
    if curvature_small(i) == 0 % Avoid division by zero
        force_pred_small(i) = current_force_pred_small - beta_small_const; % Or other appropriate handling
    end
    if force_pred_small(i) < 0
        force_pred_small(i) = 0;
    end
end
disp("Small Bending data processed.");
disp(" ");

% --- Process Large Bending Experiment Data ---
disp("Loading and processing Large Bending data...");
data_large_raw = load("data/L_10cm_OD_1.5mm/characterization_and_scale_force/force_large_bending/cv_output.csv");

% Pressure [Bar] to [Pa]
pressure_bar_large = data_large_raw(:,1); % [Bar]
pressure_mpa_large = pressure_bar_large*0.1; % [MPa]
pressure_large = pressure_mpa_large*1e6; % [Pa]

% Force [g] to [N]
force_g_large = data_large_raw(:,2); % grams
force_data_large = force_g_large*0.00980665; % [N]

% Pixel to meter conversion
arc_length_px_large = data_large_raw(:,5);
arc_length_rest_large = min(arc_length_px_large);
conv_rate_large = L/arc_length_rest_large; % [m/px]

arc_length_large = arc_length_px_large*conv_rate_large; % [m]
radius_px_large = data_large_raw(:,3);
radius_large = radius_px_large*conv_rate_large; % [m]
curvature_large = 1 ./ radius_large; % [1/m] % Handle potential division by zero
curvature_large(isinf(curvature_large)) = 0; % Or some other appropriate handling

% Force model prediction for Large Bending
disp("Calculating force model prediction for Large Bending...");
fx_pred_large = zeros(size(pressure_large));
fy_pred_large = zeros(size(pressure_large));
force_pred_large = zeros(size(pressure_large));
fy_epsilon_large_arr = zeros(size(pressure_large)); % Renamed

min_arc_length_large = min(arc_length_large);

for i = 1:length(pressure_large)
    [fx_pred_large(i), fy_pred_large(i)] = force_model(arc_length_large(i), curvature_large(i), pressure_large(i));
    
    fx_pred_large(i) = -fx_pred_large(i);
    fy_pred_large(i) = -fy_pred_large(i);
    
    current_force_pred_large = fy_pred_large(i); % Assuming prediction is in base frame

    f_epsilon_large = force_model_epsilon(arc_length_large(i), min_arc_length_large, pressure_large(i));
    f_epsilon_large = -f_epsilon_large;
    fy_epsilon_large = f_epsilon_large * sin(arc_length_large(i)*curvature_large(i));
    fy_epsilon_large_arr(i) = fy_epsilon_large; % Store

    alpha_large_const = 1e-5; % Renamed
    beta_large_const = 8.2e-3; % Renamed
    force_pred_large(i) = current_force_pred_large + alpha_large_const*pressure_large(i)*fy_epsilon_large/curvature_large(i) - beta_large_const;
    if curvature_large(i) == 0 % Avoid division by zero
        force_pred_large(i) = current_force_pred_large - beta_large_const; % Or other appropriate handling
    end
    if force_pred_large(i) < 0
        force_pred_large(i) = 0;
    end
end
disp("Large Bending data processed.");
disp(" ");


% --- Process New Data 1 ---
disp("Loading and processing New Data 1...");
new_data_file_path = "data/L_10cm_OD_1.5mm/characterization_and_scale_force/new_data/cv_output.csv";
data_new1_raw = load(new_data_file_path);


% Pressure [Bar] to [Pa]
pressure_bar_new1 = data_new1_raw(:,1); % [Bar]
pressure_mpa_new1 = pressure_bar_new1*0.1; % [MPa]
pressure_new1 = pressure_mpa_new1*1e6; % [Pa]

% Force [g] to [N]
force_g_new1 = data_new1_raw(:,2); % grams
force_data_new1 = force_g_new1*0.00980665; % [N]

% Pixel to meter conversion
arc_length_px_new1 = data_new1_raw(:,5);
arc_length_rest_new1 = min(arc_length_px_new1); % Assuming the first entry is rest for new data, or adjust as needed
conv_rate_new1 = L/arc_length_rest_new1; % [m/px]

arc_length_new1 = arc_length_px_new1*conv_rate_new1; % [m]
radius_px_new1 = data_new1_raw(:,3);
radius_new1 = radius_px_new1*conv_rate_new1; % [m]
curvature_new1 = 1 ./ radius_new1; % [1/m]
curvature_new1(isinf(curvature_new1) | isnan(curvature_new1)) = 0; % Handle Inf/NaN

% Force model prediction for New Data 1
disp("Calculating force model prediction for New Data 1...");
fx_pred_new1 = zeros(size(pressure_new1));
fy_pred_new1 = zeros(size(pressure_new1));
force_pred_new1 = zeros(size(pressure_new1));
fy_epsilon_new1_arr = zeros(size(pressure_new1));

min_arc_length_new1 = min(arc_length_new1);

for i = 1:length(pressure_new1)
    [fx_pred_new1(i), fy_pred_new1(i)] = force_model(arc_length_new1(i), curvature_new1(i), pressure_new1(i));
    
    fx_pred_new1(i) = -fx_pred_new1(i);
    fy_pred_new1(i) = -fy_pred_new1(i);
    
    current_force_pred_new1 = fy_pred_new1(i);

    f_epsilon_new1 = force_model_epsilon(arc_length_new1(i), min_arc_length_new1, pressure_new1(i));
    f_epsilon_new1 = -f_epsilon_new1;
    fy_epsilon_new1 = f_epsilon_new1 * sin(arc_length_new1(i)*curvature_new1(i));
    fy_epsilon_new1_arr(i) = fy_epsilon_new1;

    % Using small bending parameters as a placeholder for New Data 1. Adjust if needed.
    alpha_new1_const = alpha_small_const; 
    beta_new1_const = beta_small_const;
    force_pred_new1(i) = current_force_pred_new1 + alpha_new1_const*pressure_new1(i)*fy_epsilon_new1/curvature_new1(i) - beta_new1_const;
    if curvature_new1(i) == 0
        force_pred_new1(i) = current_force_pred_new1 - beta_new1_const;
    end
    if force_pred_new1(i) < 0
        force_pred_new1(i) = 0;
    end
end
disp("New Data 1 processed.");
disp(" ");


% --- Plotting ---
disp("Plotting results...");
figure
hold on

% Define text offset for average curvature labels
text_y_base_offset_factor = 1.12; 
additional_vertical_lift = 1.2; 
avg_curvature_text_fontsize = 22; 
shade_alpha = 0.2; % Transparency for the shaded variance area

% Colors (consistent with MATLAB defaults used)
color_small = [0 0.4470 0.7410]; % Blue
color_large = [0.8500 0.3250 0.0980]; % Orange/Red
color_new1  = [0.4660 0.6740 0.1880]; % Green

% Convert to plotting units (MPa for pressure, mN for force)
pressure_small_mpa = pressure_small./1e6;
force_data_small_mn = force_data_small.*1e3;
force_pred_small_mn = force_pred_small.*1e3;

pressure_large_mpa = pressure_large./1e6;
force_data_large_mn = force_data_large.*1e3;
force_pred_large_mn = force_pred_large.*1e3;

pressure_new1_mpa = pressure_new1./1e6;
force_data_new1_mn = force_data_new1.*1e3;
force_pred_new1_mn = force_pred_new1.*1e3;

% --- Determine overall pressure range for extending fit lines ---
all_pressures_mpa = [pressure_small_mpa(:); pressure_large_mpa(:); pressure_new1_mpa(:)];
% Filter out NaNs or Infs before min/max if they might occur from conversions
valid_pressures = all_pressures_mpa(all_pressures_mpa > 0 & ~isinf(all_pressures_mpa) & ~isnan(all_pressures_mpa));
if isempty(valid_pressures)
    min_pressure_plot = 0.1; % Fallback
    max_pressure_plot = 1.0; % Fallback
else
    min_pressure_plot = min(valid_pressures);
    max_pressure_plot = max(valid_pressures);
end
plot_pressure_range = linspace(min_pressure_plot * 0.90, max_pressure_plot * 1.05, 200);
x_fill_area = [plot_pressure_range, fliplr(plot_pressure_range)];

% --- Small Bending ---
% Fit for data and get error estimation structure S
[p_data_small, S_data_small] = polyfit(pressure_small_mpa, force_data_small_mn, 1);
% Get fit line and delta (error estimate for prediction interval)
[force_fit_data_small_extended, delta_data_small] = polyval(p_data_small, plot_pressure_range, S_data_small);
% Plot variable-width variance shade for data (approx 95% prediction interval)
y_fill_data_small = [force_fit_data_small_extended + 2*delta_data_small, fliplr(force_fit_data_small_extended - 2*delta_data_small)];
fill(x_fill_area, y_fill_data_small, color_small, 'FaceAlpha', shade_alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
% Fit for prediction (no shade for prediction line)
p_pred_small = polyfit(pressure_small_mpa, force_pred_small_mn, 1);
force_fit_pred_small_extended = polyval(p_pred_small, plot_pressure_range);

% --- Large Bending ---
[p_data_large, S_data_large] = polyfit(pressure_large_mpa, force_data_large_mn, 1);
[force_fit_data_large_extended, delta_data_large] = polyval(p_data_large, plot_pressure_range, S_data_large);
y_fill_data_large = [force_fit_data_large_extended + 2*delta_data_large, fliplr(force_fit_data_large_extended - 2*delta_data_large)];
fill(x_fill_area, y_fill_data_large, color_large, 'FaceAlpha', shade_alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
p_pred_large = polyfit(pressure_large_mpa, force_pred_large_mn, 1);
force_fit_pred_large_extended = polyval(p_pred_large, plot_pressure_range);

% --- New Data 1 ---
[p_data_new1, S_data_new1] = polyfit(pressure_new1_mpa, force_data_new1_mn, 1);
[force_fit_data_new1_extended, delta_data_new1] = polyval(p_data_new1, plot_pressure_range, S_data_new1);
y_fill_data_new1 = [force_fit_data_new1_extended + 2*delta_data_new1, fliplr(force_fit_data_new1_extended - 2*delta_data_new1)];
fill(x_fill_area, y_fill_data_new1, color_new1, 'FaceAlpha', shade_alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
p_pred_new1 = polyfit(pressure_new1_mpa, force_pred_new1_mn, 1);
force_fit_pred_new1_extended = polyval(p_pred_new1, plot_pressure_range);


% --- Plot Original Data Points (after shades) ---
plot(pressure_small_mpa, force_pred_small_mn, 'o', 'MarkerFaceColor', color_small, 'MarkerEdgeColor', color_small, 'MarkerSize', 5, 'HandleVisibility', 'off'); 
plot(pressure_small_mpa, force_data_small_mn, 'x', 'MarkerEdgeColor', color_small, 'MarkerSize', 7, 'LineWidth', 1.5, 'HandleVisibility', 'off'); 

plot(pressure_large_mpa, force_pred_large_mn, 's', 'MarkerFaceColor', color_large, 'MarkerEdgeColor', color_large, 'MarkerSize', 5, 'HandleVisibility', 'off');
plot(pressure_large_mpa, force_data_large_mn, '+', 'MarkerEdgeColor', color_large, 'MarkerSize', 7, 'LineWidth', 1.5, 'HandleVisibility', 'off');

plot(pressure_new1_mpa, force_pred_new1_mn, 'd', 'MarkerFaceColor', color_new1, 'MarkerEdgeColor', color_new1, 'MarkerSize', 5, 'HandleVisibility', 'off');
plot(pressure_new1_mpa, force_data_new1_mn, '*', 'MarkerEdgeColor', color_new1, 'MarkerSize', 7, 'LineWidth', 1.5, 'HandleVisibility', 'off');

% --- Plot Fit Lines (after shades and points) ---
% Small Bending Lines
plot(plot_pressure_range, force_fit_pred_small_extended, '-', 'LineWidth', 2.5, 'Color', color_small, 'HandleVisibility', 'off'); 
plot(plot_pressure_range, force_fit_data_small_extended, '--', 'LineWidth', 2.5, 'Color', color_small, 'HandleVisibility', 'off');
% Large Bending Lines
plot(plot_pressure_range, force_fit_pred_large_extended, '-', 'LineWidth', 2.5, 'Color', color_large, 'HandleVisibility', 'off'); 
plot(plot_pressure_range, force_fit_data_large_extended, '--', 'LineWidth', 2.5, 'Color', color_large, 'HandleVisibility', 'off');
% New Data 1 Lines
plot(plot_pressure_range, force_fit_pred_new1_extended, '-', 'LineWidth', 2.5, 'Color', color_new1, 'HandleVisibility', 'off'); 
plot(plot_pressure_range, force_fit_data_new1_extended, '--', 'LineWidth', 2.5, 'Color', color_new1, 'HandleVisibility', 'off'); 

% --- Add Curvature Text (after lines) ---
% Small Bending Text
avg_curvature_small = mean(curvature_small(~isinf(curvature_small) & ~isnan(curvature_small) & curvature_small~=0));
text_x_small_idx = round(length(plot_pressure_range) * 0.80); 
text_x_small = plot_pressure_range(text_x_small_idx);
text_y_small_on_line = force_fit_data_small_extended(text_x_small_idx);
text_y_small = text_y_small_on_line * text_y_base_offset_factor + additional_vertical_lift; 
if ~isnan(avg_curvature_small)
    text(text_x_small, text_y_small, sprintf('k=%.1f', avg_curvature_small), ...
         'FontSize', avg_curvature_text_fontsize, 'Color', color_small, 'Interpreter', 'tex', 'FontWeight', 'bold');
end
% Large Bending Text
avg_curvature_large = mean(curvature_large(~isinf(curvature_large) & ~isnan(curvature_large) & curvature_large~=0));
text_x_large_idx = round(length(plot_pressure_range) * 0.85); 
text_x_large = plot_pressure_range(text_x_large_idx);
text_y_large_on_line = force_fit_data_large_extended(text_x_large_idx);
text_y_large = text_y_large_on_line * text_y_base_offset_factor + additional_vertical_lift; 
if ~isnan(avg_curvature_large)
    text(text_x_large, text_y_large, sprintf('k=%.1f', avg_curvature_large), ...
         'FontSize', avg_curvature_text_fontsize, 'Color', color_large, 'Interpreter', 'tex', 'FontWeight', 'bold');
end
% New Data 1 Text
avg_curvature_new1 = mean(curvature_new1(~isinf(curvature_new1) & ~isnan(curvature_new1) & curvature_new1~=0)); 
if ~isempty(pressure_new1_mpa) && ~isnan(avg_curvature_new1)
    text_x_new1_anchor_pressure_idx = max(1,round(length(pressure_new1_mpa)*0.3)); 
    text_x_new1_anchor_pressure = pressure_new1_mpa(text_x_new1_anchor_pressure_idx);
    text_y_new1_on_line = polyval(p_data_new1, text_x_new1_anchor_pressure); % Use p_data_new1 for y-positioning
    text_y_new1 = text_y_new1_on_line * text_y_base_offset_factor + additional_vertical_lift + 0.5; 
    text_x_new1 = text_x_new1_anchor_pressure * 0.98; 
    current_ylim = ylim;
    if text_y_new1 > current_ylim(2) * 0.90 
        text_y_new1 = current_ylim(2) * 0.90;
    end
    if text_y_new1 < current_ylim(1) + (current_ylim(2)-current_ylim(1))*0.05 
         text_y_new1 = current_ylim(1) + (current_ylim(2)-current_ylim(1))*0.05;
    end
    text(text_x_new1, text_y_new1, sprintf('k=%.1f', avg_curvature_new1), ...
         'FontSize', avg_curvature_text_fontsize, 'Color', color_new1, 'Interpreter', 'tex', 'FontWeight', 'bold');
end

% --- Legend Proxy Artists ---
h_leg = gobjects(5,1); % Increased size for the new legend entry
h_leg(1) = plot(NaN, NaN, 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 7, 'DisplayName', 'Model Prediction'); 
h_leg(2) = plot(NaN, NaN, 'x', 'MarkerEdgeColor', 'k', 'MarkerSize', 9, 'LineWidth', 2, 'DisplayName', 'Measured Force'); 
h_leg(3) = plot(NaN, NaN, '-', 'Color', 'k', 'LineWidth', 2.5, 'DisplayName', 'Prediction Best Fit');
h_leg(4) = plot(NaN, NaN, '--', 'Color', 'k', 'LineWidth', 2.5, 'DisplayName', 'Data Best Fit');
% Proxy for the shaded area
h_leg(5) = patch(NaN, NaN, 'k', 'FaceAlpha', shade_alpha, 'EdgeColor', 'none', 'DisplayName', '~95% Prediction Interval');


% --- Plot Formatting ---
xlabel('Pressure (MPa)', 'Interpreter', 'latex', 'fontsize', 38); 
ylabel('Force (mN)', 'Interpreter', 'latex', 'fontsize', 38); 
legend(h_leg, 'Location', 'NorthWest', 'FontSize', 16, 'Box', 'off'); 
title('Force Prediction vs Data', 'Interpreter', 'latex', 'fontsize', 36); 

ax = gca;
ax.FontSize = 30; 
grid on;
box on; 

% --- Axis limits ---
current_xlim_auto = xlim; 
xlim_lower_candidate = min_pressure_plot * 0.9;
if isnan(xlim_lower_candidate) || isinf(xlim_lower_candidate) || xlim_lower_candidate <=0 || isempty(xlim_lower_candidate)
    xlim_lower_final = current_xlim_auto(1); 
else
    xlim_lower_final = xlim_lower_candidate;
end
xlim_upper_candidate = max_pressure_plot * 1.1;
if isnan(xlim_upper_candidate) || isinf(xlim_upper_candidate) || isempty(xlim_upper_candidate)
    xlim_upper_final = current_xlim_auto(2);
else
    xlim_upper_final = xlim_upper_candidate;
end
if xlim_lower_final < xlim_upper_final
    xlim([xlim_lower_final, xlim_upper_final]);
else
    xlim(current_xlim_auto); 
end

ylim_lower_val = 0; 
ylim_upper_fixed = 10; 
current_ylim_auto = ylim; 

all_forces_for_ylim_check = [force_data_small_mn(:); force_pred_small_mn(:); force_data_large_mn(:); force_pred_large_mn(:); force_data_new1_mn(:); force_pred_new1_mn(:)];
min_force_actual_data = min(all_forces_for_ylim_check(~isnan(all_forces_for_ylim_check) & ~isinf(all_forces_for_ylim_check)));

if ~isempty(min_force_actual_data) && min_force_actual_data < 0
    ylim_lower_val = min_force_actual_data * 1.1; 
end

if ylim_lower_val >= ylim_upper_fixed 
    ylim_upper_final_val = ylim_lower_val + 1; 
else
    ylim_upper_final_val = ylim_upper_fixed;
end 
ylim([ylim_lower_val, ylim_upper_final_val]);


hold off;

disp("Plotting complete.");

% --- RMSE and Error Standard Deviation Calculation ---
disp(" "); % Add a blank line for better readability in terminal
disp("Calculating Overall RMSE and Standard Deviation of Prediction Errors...");

% Ensure all data is in the same units (mN as used in plotting)
% These should already be defined from the plotting section:
% force_data_small_mn, force_pred_small_mn
% force_data_large_mn, force_pred_large_mn
% force_data_new1_mn, force_pred_new1_mn

if exist('force_data_small_mn', 'var') && exist('force_pred_small_mn', 'var') && ...
   exist('force_data_large_mn', 'var') && exist('force_pred_large_mn', 'var') && ...
   exist('force_data_new1_mn', 'var') && exist('force_pred_new1_mn', 'var')

    % Calculate prediction errors (residuals) for each dataset
    errors_small = force_pred_small_mn - force_data_small_mn;
    errors_large = force_pred_large_mn - force_data_large_mn;
    errors_new1  = force_pred_new1_mn  - force_data_new1_mn;

    % Concatenate all errors
    all_errors = [errors_small(:); errors_large(:); errors_new1(:)];

    % Calculate squared errors for RMSE
    all_squared_errors = all_errors.^2;

    % Calculate Mean Squared Error (MSE)
    mean_squared_error = mean(all_squared_errors);

    % Calculate Root Mean Squared Error (RMSE)
    overall_rmse = sqrt(mean_squared_error);

    % Calculate Standard Deviation of the prediction errors
    std_dev_errors = std(all_errors);

    fprintf('Overall RMSE (Data vs. Prediction): %.4f mN\n', overall_rmse);
    fprintf('Standard Deviation of Prediction Errors: %.4f mN\n', std_dev_errors);

else
    disp("RMSE and Error StdDev calculation skipped: Not all required force data/prediction variables are available.");
    disp("Ensure all three datasets (small, large, new1) are processed.");
end
disp(" ");
disp("Script finished.");