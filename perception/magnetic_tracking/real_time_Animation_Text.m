% real_time_Animation_Text:  Implements "real-time" data acquisition (Note: 
% we are using synchronous mode) and plotting (called from pull down menu)
% Ascension Technology Corporation 


global libstring;
global SensorNumAttached; % Attached sensors number
global hMainfig;% Main figure
global tracker_run;    % starts and stops I/O
global numBoards;% number of boards
global hp;
global xstart;
global xstep;
%global ystart;  will be set based on window size below
global ystep;
global height;
global width;
global sensorwidth; % width of sensor label column

clf;% clear figure

NumSenPres = sum(SensorNumAttached);

% hMainfig;
senSorPre = ones(1, 4*numBoards);
TRUE=1;FALSE=0;ON=1;OFF=0;
X=1;Y=2;Z=3;

tracker_run = OFF;    % set to stop during setup

% Create Pull Down Menu to Start/Stop I?O if not yet up
if isempty(findobj(gcf,'Label','Run tracker'))
    % Create a new a menu on the menu bar at the top of the figure window
    % with the handle hfig
    h_menu = uimenu(gcf,'Label','Run tracker');
    
    % submenus are added by specfiying the handle to the column menu
    uimenu(h_menu,'Label','Start', ...
        'Callback', 'real_time_plot_text;');
    uimenu(h_menu,'Label','Stop', ...
        'Callback', 'tracker_run=0;');
end

xstart = 5;
xstep = 80;
% ystart is not global
ystep = 30;
height = 25;
width = 70;
sensorwidth = 80; % width of sensor label column

% creates a uipanel to display sensor data
hp   = uipanel('parent',hMainfig, 'BorderType', 'line', ...
                'ResizeFcn', 'data_resize',...
                'Units','normalized',...
                'Position',[0,0,1,1]);

set(hp, 'Units', 'pixels');
p = get(hp,'Position');
set(hp, 'Units', 'normalized');
ystart = p(2)+p(4)-40;
       
% headers
pos = [xstart,ystart,width,height];
hbss = uicontrol('Style', 'text','Parent',hp,'String','sensor', ...
                 'fontSize', 16,...
                 'Units','pixels',...
                 'Position',pos);
             
pos = [xstart+sensorwidth,ystart,width,height];
hbsx = uicontrol('Style', 'text','Parent',hp,'String','x', ...
                 'fontSize', 16,...
                 'Units','pixels',...
                 'Position',pos);
             
pos = pos + [xstep,0,0,0];
hbsy = uicontrol('Style', 'text','Parent',hp,'String','y', ...
                 'fontSize', 16,...                 
                 'Units','pixels',...
                 'Position',pos);

pos = pos + [xstep,0,0,0];
hbsz = uicontrol('Style', 'text','Parent',hp,'String','z', ...
                 'fontSize', 16,...                 
                 'Units','pixels',...
                 'Position',pos);

pos = pos + [xstep,0,0,0];
hbsa = uicontrol('Style', 'text','Parent',hp,'String','az', ...
                 'fontSize', 16,...                 
                 'Units','pixels',...
                 'Position',pos);

pos = pos + [xstep,0,0,0];
hbse = uicontrol('Style', 'text','Parent',hp,'String','el', ...
                 'fontSize', 16,...                 
                 'Units','pixels',...
                 'Position',pos);

pos = pos + [xstep,0,0,0];
hbsr = uicontrol('Style', 'text','Parent',hp,'String','rl', ...
                 'fontSize', 16,...                 
                 'Units','pixels',...
                 'Position',pos);

pos = pos + [xstep,0,0,0];
hbsq = uicontrol('Style', 'text','Parent',hp,'String','q', ...
                 'fontSize', 16,...                 
                 'Units','pixels',...
                 'Position',pos);

pos = [xstart, ystart-ystep, sensorwidth, height];
for kk = 1: 4 * numBoards ,
    tempname = num2str(kk);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String',tempname,...
             'fontSize', 16,...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbs' num2str(kk) '= temp' ]);  
    temppos = getpixelposition(temp);
    evalc(['pos' num2str(kk)  ' = temppos' ]); 
    pos = pos + [0, -ystep, 0, 0];
             
end


for kk =1:4 * numBoards
    evalc(['s' num2str(kk) '=  num2str(0)']); 
    evalc(['xs' num2str(kk) '=  num2str(0)']); 
    evalc(['ys' num2str(kk) '=  num2str(0)']); 
    evalc(['zs' num2str(kk) '=  num2str(0)']); 
    evalc(['as' num2str(kk) '=  num2str(0)' ]); 
    evalc(['es' num2str(kk) '=  num2str(0)']); 
    evalc(['rs' num2str(kk) '=  num2str(0)']); 
    evalc(['qs' num2str(kk) '=  num2str(0)']); 
end

% x-position
pos = [xstart+sensorwidth, ystart-ystep, width, height];
for kk = 1: 4 * numBoards ,
    tempname = (['x'  num2str(kk) ]);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String', tempname,...
             'fontSize', 12,...
             'BackgroundColor',[1,1,1],...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbsx' num2str(kk) '= temp' ]);  
    pos = pos + [0, -ystep, 0, 0];
end

% y-position
pos = [xstart+sensorwidth+xstep, ystart-ystep, width, height];
for kk = 1: 4 * numBoards ,
    tempname = (['y'  num2str(kk) ]);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String', tempname,...
             'fontSize', 12,...
             'BackgroundColor',[1,1,1],...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbsy' num2str(kk) '= temp' ]);  
    pos = pos + [0, -ystep, 0, 0];
end

% z-position
pos = [xstart+sensorwidth+2*xstep, ystart-ystep, width, height];
for kk = 1: 4 * numBoards ,
    tempname = (['z'  num2str(kk) ]);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String', tempname,...
             'fontSize', 12,...
             'BackgroundColor',[1,1,1],...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbsz' num2str(kk) '= temp' ]);  
    pos = pos + [0, -ystep, 0, 0];
end


% azimuth data
pos = [xstart+sensorwidth+3*xstep, ystart-ystep, width, height];
for kk = 1: 4 * numBoards ,
    tempname = (['a'  num2str(kk) ]);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String', tempname,...
             'fontSize', 12,...
             'BackgroundColor',[1,1,1],...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbsa' num2str(kk) '= temp' ]);  
    pos = pos + [0, -ystep, 0, 0];
end

% elevation data
pos = [xstart+sensorwidth+4*xstep, ystart-ystep, width, height];
for kk = 1: 4 * numBoards ,
    tempname = (['e'  num2str(kk) ]);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String', tempname,...
             'fontSize', 12,...
             'BackgroundColor',[1,1,1],...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbse' num2str(kk) '= temp' ]);  
    pos = pos + [0, -ystep, 0, 0];
end

% roll data
pos = [xstart+sensorwidth+5*xstep, ystart-ystep, width, height];
for kk = 1: 4 * numBoards ,
    tempname = (['r'  num2str(kk) ]);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String', tempname,...
             'fontSize', 12,...
             'BackgroundColor',[1,1,1],...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbsr' num2str(kk) '= temp' ]);  
    pos = pos + [0, -ystep, 0, 0];
end

% quality data
pos = [xstart+sensorwidth+6*xstep, ystart-ystep, width, height];
for kk = 1: 4 * numBoards ,
    tempname = (['q'  num2str(kk) ]);
    temp = uicontrol('Style', 'text','Parent',hp,...
    'String', tempname,...
             'fontSize', 12,...
             'BackgroundColor',[1,1,1],...
             'Units','pixels',...
             'Position', pos);
    evalc(['hbsq' num2str(kk) '= temp' ]);  
    pos = pos + [0, -ystep, 0, 0];
end

% Initialize sensors data to zero
for kk = 1: 4 * numBoards ,
    set(eval(['hbsx' num2str(kk)]), 'String', num2str(eval(['xs' num2str(kk)])) );
    set(eval(['hbsy' num2str(kk)]), 'String', num2str(eval(['ys' num2str(kk)])) );
    set(eval(['hbsz' num2str(kk)]), 'String', num2str(eval(['zs' num2str(kk)])) );
    set(eval(['hbsa' num2str(kk)]), 'String', num2str(eval(['as' num2str(kk)])) );
    set(eval(['hbse' num2str(kk)]), 'String', num2str(eval(['es' num2str(kk)])) );
    set(eval(['hbsr' num2str(kk)]), 'String', num2str(eval(['es' num2str(kk)])) );
    set(eval(['hbsq' num2str(kk)]), 'String', num2str(eval(['qs' num2str(kk)])) );    
end


% Setup Handlge Graphics for high speed animation
vec_length = 1;  % length of vector on plot showing tracker axes

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

% Define an array of colors for each tracker
colorstr_ar = [{'r'} {'b'} {'g'} {'m'} {'y'} {'c'} ];
while length(colorstr_ar)<NumSenPres  
    colorstr_ar = [colorstr_ar colorstr_ar]; % repeat colors if array is not long enough
end

%Get synchronous Record
Error   = calllib(libstring, 'GetSynchronousRecord',  hex2dec('ffff'), pRecord, 4*numBoards*64);
errorHandler(Error);
Record = get(pRecord, 'Value');
for count = 1:4*numBoards
    Pos(1, count) = Record.(['x' num2str(count - 1)]);
    Pos(2, count) = Record.(['y' num2str(count - 1)]);
    Pos(3, count) = Record.(['z' num2str(count - 1)]);
    Ang(1, count) = Record.(['a' num2str(count - 1)]);
    Ang(2, count) = Record.(['e' num2str(count - 1)]);
    Ang(3, count) = Record.(['r' num2str(count - 1)]);
    time( count) = Record.(['time' num2str(count - 1)]);        
    Quality( count) = Record.(['quality' num2str(count - 1)]); 
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
    else
            senSorPre(count) = 0;                      
    end
end
indsenSorPre =  find(senSorPre == 1);

% Free memory
clear  pRecord;


