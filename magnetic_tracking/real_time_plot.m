% function real_time_plot animates tracker position and axes after data has
% been acquired 

% Ascension Technology Corporation 
global hMainfig;
global SensorNumAttached;
global hp;

NumSenPres = sum(SensorNumAttached);
TRUE=1;FALSE=0;ON=1;OFF=0;

tracker_run = ON;  % global variable used in pull down menu to start and stop data acquisition
% Get asynchronous Record
Record  = libstruct('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD');  
pRecord = libpointer('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD', Record);

while tracker_run == ON
   for count = 1:4
 %       Counter = Counter + 1;
        if SensorNumAttached(count)
                % Get asynchronous Record
                Error   = calllib('ATC3DG', 'GetAsynchronousRecord', count - 1, pRecord, 64);
                errorHandler(Error);
                
                Pos(count, 1)   = Record.x;
                Pos(count, 2)   = Record.y;
                Pos(count, 3)   = Record.z;
                Ang(count, 1)   = Record.a;
                Ang(count, 2)   = Record.e;
                Ang(count, 3)   = Record.r; 
                Quality(count)  = Record.quality;  
                
                   
            if (count == 1), 
                set(hbsx1, 'String', num2str(Record.x, '%2.2f') );
                set(hbsy1, 'String', num2str(Record.y, '%2.2f') );
                set(hbsz1, 'String', num2str(Record.z, '%2.2f') );
                set(hbsa1, 'String', num2str(Record.a, '%3.2f') );
                set(hbse1, 'String', num2str(Record.e, '%3.2f') );
                set(hbsr1, 'String', num2str(Record.r, '%3.2f') );
                set(hbsq1, 'String', num2str(Record.quality, '%4.0f') );                
            end % display sensor 1 data
            
            if (count == 2), 
                set(hbsx2, 'String', num2str(Record.x, '%2.2f') );
                set(hbsy2, 'String', num2str(Record.y, '%2.2f') );
                set(hbsz2, 'String', num2str(Record.z, '%2.2f') );
                set(hbsa2, 'String', num2str(Record.a, '%3.2f') );
                set(hbse2, 'String', num2str(Record.e, '%3.2f') );
                set(hbsr2, 'String', num2str(Record.r, '%3.2f') );
                set(hbsq2, 'String', num2str(Record.quality, '%4.0f') );                
            end % display sensor 2 data
       
            if (count == 3), 
                set(hbsx3, 'String', num2str(Record.x, '%2.2f') );
                set(hbsy3, 'String', num2str(Record.y, '%2.2f') );
                set(hbsz3, 'String', num2str(Record.z, '%2.2f') );
                set(hbsa3, 'String', num2str(Record.a, '%3.2f') );
                set(hbse3, 'String', num2str(Record.e, '%3.2f') );
                set(hbsr3, 'String', num2str(Record.r, '%3.2f') );
                set(hbsq3, 'String', num2str(Record.quality, '%4.0f') );                
            end % display sensor 3 data

            if (count == 4), 
                set(hbsx4, 'String', num2str(Record.x, '%2.2f') );
                set(hbsy4, 'String', num2str(Record.y, '%2.2f') );
                set(hbsz4, 'String', num2str(Record.z, '%2.2f') );
                set(hbsa4, 'String', num2str(Record.a, '%3.2f') );
                set(hbse4, 'String', num2str(Record.e, '%3.2f') );
                set(hbsr4, 'String', num2str(Record.r, '%3.2f') );
                set(hbsq4, 'String', num2str(Record.quality, '%4.0f') );                
            end % display sensor 4 data
                    
  
                drawnow  % forces Matlab to evaluate Callback in figure
              
        end
   end
 
   

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
clear Record ;  
clear  pRecord;

