function [x_tip , y_tip] = kin_model(k, L)
    q = k * L;
    % Validate inputs
    if k < 1e-4
        x_tip = L;
        y_tip = 0;        
    else
        x_tip = L * sin(q) / q;
        y_tip = L * ( 1 - cos(q) ) / q;
    end
end