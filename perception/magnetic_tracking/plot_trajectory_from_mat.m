function plot_trajectory_from_mat(matFileName)
% plot_trajectory_from_mat - Plots 3D trajectory from a .mat file
%
% INPUT:
%   matFileName - name of the .mat file containing variable 'temp'
%                 (with 8 columns: x y z a e r timestamp quality)
%
% OUTPUT:
%   3D trajectory plot

    % Load .mat file
    data = load(matFileName);

    if ~isfield(data, 'temp')
        error('The file does not contain variable ''temp''.');
    end

    pos = data.temp(:, 1:3); % Extract x, y, z columns

    % Plot trajectory
    figure;
    plot3(pos(:,1), pos(:,2), pos(:,3), 'b-', 'LineWidth', 1.5);
    hold on;
    plot3(pos(1,1), pos(1,2), pos(1,3), 'go', 'MarkerSize', 10, 'DisplayName', 'Start');
    plot3(pos(end,1), pos(end,2), pos(end,3), 'ro', 'MarkerSize', 10, 'DisplayName', 'End');
    hold off;

    grid on;
    axis equal;
    xlabel('X (inches)');
    ylabel('Y (inches)');
    zlabel('Z (inches)');
    title(['3D Trajectory from ', matFileName], 'Interpreter', 'none');
    legend show;
end
