function f_epsilon = force_model_epsilon(L, L0, p)

    epsilon = (L - L0) / L0;

    % Get linear model from data processing
    load("data/free_motion/linear_model_p_vs_epsilon.mat", "mdl");
    epsilon_data = mdl.Coefficients.Estimate(2);
    epsilon_intercept = mdl.Coefficients.Estimate(1);

    %  % Get the value where the linear model intersects the x-axis (minum pressure required for elongation)
    p_intercept = -epsilon_intercept / epsilon_data;

    % If the input pressure is less than the minimum pressure required for bending, the force is zero
    if p < p_intercept
        f_epsilon = 0;
        return;
    end

    addpath(genpath('./'));

    % Call the elastic term fuction
    K = elastic_term_epsilon(epsilon);
    % Call the input force function
    u = 6.18*input_force(p); % recall the model is wrong by a factor of 6.18 wrt to the data

    % Calculate the force model
    if abs(u) > abs(K)
        force = (K - u);
    else
        % If the input torque is less than the elastic term, the force is zero because the elongation does not occur
        % Note: this is valid only for the type of experiment performed
        disp("The input force is less than the elastic term, the force is zero because the elongation does not occur");
        force =  0;
    end

    f_epsilon = force;
end