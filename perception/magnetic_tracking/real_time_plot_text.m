% function real_time_plot animates tracker position and axes after data has
% been acquired 

% Ascension Technology Corporation 
global libstring;
global hMainfig;
global SensorNumAttached;% number of attached sensors
global hp;
global numBoards;% number of boards

NumSenPres = sum(SensorNumAttached);
TRUE=1;FALSE=0;ON=1;OFF=0;

tracker_run = ON;  % global variable used in pull down menu to start and stop data acquisition
% Get synchronous Record
% Initialize structure
for kk = 0:(4 * numBoards - 1),
   evalc(['sm.x' num2str(kk) '=0']);
   evalc(['sm.y' num2str(kk) '=0']);
   evalc(['sm.z' num2str(kk) '=0']);
   evalc(['sm.a' num2str(kk) '=0']);
   evalc(['sm.e' num2str(kk) '=0']);
   evalc(['sm.r' num2str(kk) '=0']);
   evalc(['sm.time' num2str(kk) '=0']);
   evalc(['sm.quality' num2str(kk) '=0']);  
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

while tracker_run == ON
    
    % Get synchronous Record
    Error   = calllib(libstring, 'GetSynchronousRecord',  hex2dec('ffff'), pRecord, 4*numBoards*64);
    errorHandler(Error);
    Record = get(pRecord, 'Value');
    for count = 1:4*numBoards
        evalc(['Pos(1, count) ='  'Record.x' num2str(count - 1)]);
        evalc(['Pos(2, count) ='  'Record.y' num2str(count - 1)]);
        evalc(['Pos(3, count) ='  'Record.z' num2str(count - 1)]);
        evalc(['Ang(1, count) ='  'Record.a' num2str(count - 1)]);
        evalc(['Ang(2, count) ='  'Record.e' num2str(count - 1)]);
        evalc(['Ang(3, count) ='  'Record.r' num2str(count - 1)]);
        evalc(['time( count) ='  'Record.time' num2str(count - 1)]);        
        evalc(['Quality( count) ='  'Record.quality' num2str(count - 1)]); 
    end

    for count = 1:4*numBoards
        if SensorNumAttached(count)
            set(eval(['hbsx' num2str(count)]), 'String', num2str(eval(['Record.x' num2str(count - 1)]), '%2.2f') );
            set(eval(['hbsy' num2str(count)]), 'String', num2str(eval(['Record.y' num2str(count - 1)]), '%2.2f') );
            set(eval(['hbsz' num2str(count)]), 'String', num2str(eval(['Record.z' num2str(count - 1)]), '%2.2f') );
            set(eval(['hbsa' num2str(count)]), 'String', num2str(eval(['Record.a' num2str(count - 1)]), '%2.2f') );
            set(eval(['hbse' num2str(count)]), 'String', num2str(eval(['Record.e' num2str(count - 1)]), '%2.2f') );
            set(eval(['hbsr' num2str(count)]), 'String', num2str(eval(['Record.r' num2str(count - 1)]), '%2.2f') );
            set(eval(['hbsq' num2str(count)]), 'String', num2str(eval(['Record.quality' num2str(count - 1)]), '%2.2f') );
            drawnow  % forces Matlab to evaluate Callback in figure
        end
    end
 end
% Free memory
clear  pRecord;

