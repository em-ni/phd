function tracker_position_angles_Quality
% tracker_position_angles: displays attached sensors positions, angles 
% and quality number to the MATLAB command window

% Ascension Technology Corporation 

global libstring;
global SensorNumAttached; % number of attached sensors
global numBoards;% number of boards

% Get synchronous Record
% Initialize structure

for kk = 0:(4 * numBoards - 1),
   sm.(['x' num2str(kk)]) = 0;
   sm.(['y' num2str(kk)]) = 0;
   sm.(['z' num2str(kk)]) = 0;
   sm.(['a' num2str(kk)]) = 0;
   sm.(['e' num2str(kk)]) = 0;
   sm.(['r' num2str(kk)]) = 0;
   sm.(['time' num2str(kk)]) = 0;
   sm.(['quality' num2str(kk)]) = 0;  
end

if numBoards == 1
    pRecord = libpointer('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD_AllSensors_Four', sm);
elseif numBoards == 2
    pRecord = libpointer('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD_AllSensors_Eight', sm);
elseif numBoards == 3
    pRecord = libpointer('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD_AllSensors_Twelve', sm);
else 
    pRecord = libpointer('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD_AllSensors_Sixting', sm);
end
    
   
Error   = calllib(libstring, 'GetAsynchronousRecord',  hex2dec('ffff'), pRecord, 4*numBoards*64);
errorHandler(Error);

Record = get(pRecord, 'Value');
for count = 1:4*numBoards
    tempPos(1, count) = Record.(['x' num2str(count - 1)]);
    tempPos(2, count) = Record.(['y' num2str(count - 1)]);
    tempPos(3, count) = Record.(['z' num2str(count - 1)]);
    tempAng(1, count) = Record.(['a' num2str(count - 1)]);
    tempAng(2, count) = Record.(['e' num2str(count - 1)]);
    tempAng(3, count) = Record.(['r' num2str(count - 1)]);
    temptime( count) = Record.(['time' num2str(count - 1)]);        
    tempQuality( count) = Record.(['quality' num2str(count - 1)]); 
end
clf;% clear figure
for count = 1:4*numBoards
    if SensorNumAttached(count)
        % Display data at the MATLAB command window
       
        display(sprintf('\nSensor number %g position and angles in inches and degrees, respectively.', ...
                        count))
        disp(sprintf('x = %g inches', tempPos(1, count) )) 
        disp(sprintf('y = %g inches', tempPos(2, count) )) 
        disp(sprintf('z = %g inches', tempPos(3, count) )) 

        disp(sprintf('a = %g degrees', tempAng(1, count) ))
        disp(sprintf('e = %g degrees', tempAng(2, count) )) 
        disp(sprintf('r = %g degrees', tempAng(3, count) ))
        
        disp(sprintf('quality = %g ', tempQuality( count)) )

    end
end
clear sm;



