function [Pos, Ang, TimStam, Quality, Error] = tracker_stream_save(OptDispl)
% The tracker_stream_save: designed to place the tracker in a 
% stream mode. 
% input: OptDispl
%        (OptDispl = 1) => sensor position is displayed on the MATLAB
%        command window
%        (OptDispl = 0) => sensor position is not displayed
% output: Pos: Attached sensors positions
%         Ang: Attached sensors angles
%         TimStam : timestamp
%         Error: Error code
% Ascension Technology Corporation 

global libstring;
global SensorNumAttached;
global hMainfig;
global gui;
global numBoards;

clf;

TRUE=1;FALSE=0;
ON=1;OFF=0;

gui.start_proc = 0;
gui.stop_proc  = 0;

if length(findobj(gcf,'Label','Run tracker'))==1
    hMainfig('Run tracker', 'off');
end
hfig = figure('Position',[408 327 360 244],...
    'MenuBar','none', ...
    'Name','Procedure Menu',...
    'NumberTitle','off');
% Start button
hpush_start = uicontrol(hfig,'Style','pushbutton');
set(hpush_start,'Position',[20 200 150 20]); %[left bottom width height]
set(hpush_start,'String','Start Saving STREAM I/O of tracker');
set(hpush_start,'Callback','gui.start_proc=1;');

% Stop button
hpush_stop = uicontrol(hfig,'Style','pushbutton');
set(hpush_stop,'Position',[20 150 150 20]); %[left bottom width height]
set(hpush_stop,'String','Stop Saving STREAM I/O of tracker');
set(hpush_stop,'Callback','gui.stop_proc=1;');
set(hpush_stop,'Enable','off');


while (gui.start_proc == 0)
    drawnow
    %    disp('waiting to start')
end
set(hpush_start,'Enable','off');
set(hpush_stop,'Enable','on');

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
    

Counter   = 0;
senSorPre = ones(1,4*numBoards);%[1 1 1 1];
NumSenPres = sum(SensorNumAttached);


while (gui.stop_proc == 0 )
    
    clc
    % Get synchronous Record
    Error   = calllib(libstring, 'GetSynchronousRecord',  hex2dec('ffff'), pRecord, 4*numBoards*64);
    errorHandler(Error);
    Counter = Counter + 1;
    
    for count = 1:4*numBoards
        Record = get(pRecord, 'Value');
        Pos(Counter, 1, count) = Record.(['x' num2str(count-1)]);
        Pos(Counter, 2, count) = Record.(['y' num2str(count-1)]);
        Pos(Counter, 3, count) = Record.(['z' num2str(count-1)]);
        Ang(Counter, 1, count) = Record.(['a' num2str(count-1)]);
        Ang(Counter, 2, count) = Record.(['e' num2str(count-1)]);
        Ang(Counter, 3, count) = Record.(['r' num2str(count-1)]);
        time(Counter, 1, count) = Record.(['time' num2str(count-1)]);
        Quality(Counter, 1, count) = Record.(['quality' num2str(count-1)]);

        if(OptDispl == 1)
            fprintf('\nSensor number %g position in inches: \n',count)
            disp([Pos(Counter, 1, count) Pos(Counter, 2, count)...
                  Pos(Counter, 3, count)])
        end
        drawnow  % forces Matlab to evaluate Callback in figure
    end
    TimStam(Counter)  = Record.time0;
    
end
indsenSorPre =  find(SensorNumAttached == 1);



% Pos = zeros(Counter,3,4*numBoards);
% Ang = zeros(Counter,3,4*numBoards);
% time = zeros(Counter,1,4*numBoards);
% Quality = zeros(Counter,1,4*numBoards);



set(hpush_stop,'Enable','off'); % show that button has been pressed
close(hfig)
clear hfig;

% Force the tracker system to come out of STREAM mode by Issuing the 
% command GetSensorStatus
for count = 1:4*numBoards
    if SensorNumAttached(count)
        Error1 = calllib(libstring, 'GetSensorStatus', count - 1);
        errorHandler(Error1);
    end
end

drawnow;




