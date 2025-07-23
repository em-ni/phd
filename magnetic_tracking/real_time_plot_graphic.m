% function real_time_plot animates tracker position and axes after data has
% been acquired 

% Ascension Technology Corporation 
global libstring;
global hMainfig;
global SensorNumAttached;% number of attached sensors
global hp;
global numBoards; % number of boards

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
        evalc(['Pos(count, 1) ='  'Record.x' num2str(count - 1)]);
        evalc(['Pos(count, 2) ='  'Record.y' num2str(count - 1)]);
        evalc(['Pos(count, 3) ='  'Record.z' num2str(count - 1)]);
        evalc(['Ang(count, 1) ='  'Record.a' num2str(count - 1)]);
        evalc(['Ang(count, 2) ='  'Record.e' num2str(count - 1)]);
        evalc(['Ang(count, 3) ='  'Record.r' num2str(count - 1)]);
        evalc(['time( count) ='  'Record.time' num2str(count - 1)]);        
        evalc(['Quality( count) ='  'Record.quality' num2str(count - 1)]); 
    end

    for count = 1:4*numBoards
        if ~SensorNumAttached(count)
            senSorPre(count) = 0;                      
        end
    end
    indsenSorPre =  find(senSorPre == 1);


    % plot postion and axes
    for itracker=1:NumSenPres
        pos = Pos(indsenSorPre(itracker),X:Z);
        ang = Ang(indsenSorPre(itracker),X:Z) ;
        R = tracker_Rot(ang* 0.017453292519943);             % Calculate Rotation matrix R whose columsn are the unit axes of each tracker coordinate system
        xvec = pos' + R(:,X)*vec_length;  
        yvec = pos' + R(:,Y)*vec_length;
        zvec = pos' + R(:,Z)*vec_length;

        set(hreal_time_xvec(indsenSorPre(itracker)),'XData',[pos(X);xvec(X)]);
        set(hreal_time_xvec(indsenSorPre(itracker)),'YData',[pos(Y);xvec(Y)]);
        set(hreal_time_xvec(indsenSorPre(itracker)),'ZData',[pos(Z);xvec(Z)]);

        set(hreal_time_yvec(indsenSorPre(itracker)),'XData',[pos(X);yvec(X)]);
        set(hreal_time_yvec(indsenSorPre(itracker)),'YData',[pos(Y);yvec(Y)]);
        set(hreal_time_yvec(indsenSorPre(itracker)),'ZData',[pos(Z);yvec(Z)]);

        set(hreal_time_zvec(indsenSorPre(itracker)),'XData',[pos(X);zvec(X)]);
        set(hreal_time_zvec(indsenSorPre(itracker)),'YData',[pos(Y);zvec(Y)]);
        set(hreal_time_zvec(indsenSorPre(itracker)),'ZData',[pos(Z);zvec(Z)]);
    
        % set texdt position
        set(hreal_time_xtext(indsenSorPre(itracker)),'Position',xvec);
        set(hreal_time_ytext(indsenSorPre(itracker)),'Position',yvec);
        set(hreal_time_ztext(indsenSorPre(itracker)),'Position',zvec); 
        drawnow
    end        

    drawnow
%     pause(0.05)
 end
% Free memory
clear  pRecord;

