function [f_x, f_y] = force_model(L, k, p)

    % Get linear model from data_processing
    load("data/L_10cm_OD_1.5mm/characterization_and_scale_force/free_motion/linear_model_p_vs_k.mat", "mdl");
    k_data = mdl.Coefficients.Estimate(2);
    k_intercept = mdl.Coefficients.Estimate(1);

    % Get the value where the linear model intersects the x-axis (minum pressure required for bending)
    p_intercept = -k_intercept / k_data;

    % If the input pressure is less than the minimum pressure required for bending, the force is zero
    if p < p_intercept
        f_x = 0;
        f_y = 0;
        return;
    end

    q = k * L;
    % Validate inputs
    if abs(k) < 1e-4
        error('q cannot be zero because it is used as a denominator');
        % I should write the limit here for the sake of completeness
    end

    % Jacobian matrix
    J = [L * cos(q) / q - L * sin(q) / q^2 ; L * sin(q)];
    % Pseudo-inverse of the Jacobian
    J_pinv = pinv(J);

    addpath(genpath('./'));

    % Call the elastic term function
    K = elastic_term(k);
    % Call the input torque function
    u = input_torque(p);

    % Calculate the force model
    if abs(u) > abs(K)
        force = J_pinv * (K - u);
    else
        % If the input torque is less than the elastic term, the force is zero because the bending does not occur
        % Note: this is valid only for the type of experiment performed
        disp("The input torque is less than the elastic term, the force is zero because the bending does not occur");
        force = J_pinv * 0;
    end
    

    % Output force components (in ??? frame)
    f_x = force(1);
    f_y = force(2);
end
