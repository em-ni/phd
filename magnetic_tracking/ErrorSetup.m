% ErrorSetup: display an error message to the MATLAB command window

% Ascension Technology Corporation 
function ErrorSetup(y)
if (y == 0)
    close all;
    error('System was not initialized, please click "Setup tracker" on the "Run tracker" menu to initialize the system');
    
end