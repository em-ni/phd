% real_time_Animation:  Implements "real-time" data acquisition (Note: 
% we are using asynchronous mode) and plotting (called from pull down menu)
% Ascension Technology Corporation 


global SensorNumAttached; % Attached sensors number
global hMainfig;% Main figure
global tracker_run;    % starts and stops I/O
global hp;

clf;% clear figure
axes('position',[.1  .1  .8  .6])
view(3)
axis vis3d  % set axis for proper 3D visulization
setup_views    % load pull downmenu for pre defined views

NumSenPres = sum(SensorNumAttached);

% hMainfig;
senSorPre = [1 1 1 1];
TRUE=1;FALSE=0;ON=1;OFF=0;
X=1;Y=2;Z=3;

tracker_run = OFF;    % set to stop during setup
view(-37.5,30)
axis vis3d  % set axis for proper 3D visulization


% Create Pull Down Menu to Start/Stop I?O if not yet up
if isempty(findobj(gcf,'Label','Run tracker'))
    % Create a new a menu on the menu bar at the top of the figure window
    % with the handle hfig
    h_menu = uimenu(gcf,'Label','Run tracker');
    
    % submenus are added by specfiying the handle to the column menu
    uimenu(h_menu,'Label','Start', ...
        'Callback', 'real_time_plot;');
    uimenu(h_menu,'Label','Stop', ...
        'Callback', 'tracker_run=0;');
end
% uimenu(hMainfig,'Label','Run tracker','Enable','on');
title('Start Real Time tracker I/O from "Run tracker" Pull Down Menu',...
    'FontWeight','Bold')
% define labels and titles
xlabel('X');
ylabel('Y');
zlabel('Z');

axis([5 25 -12 12 -15 15]);
axis square;

s1  =num2str(0); s2  =num2str(0); s3  =num2str(0);s4  =num2str(0);
xs1 =num2str(0); xs2 =num2str(0); xs3 =num2str(0);xs4 =num2str(0);
ys1 =num2str(0); ys2 =num2str(0); ys3 =num2str(0);ys4 =num2str(0);
zs1 =num2str(0); zs2 =num2str(0); zs3 =num2str(0);zs4 =num2str(0);
as1 =num2str(0); as2 =num2str(0); as3 =num2str(0);as4 =num2str(0);
es1 =num2str(0); es2 =num2str(0); es3 =num2str(0);es4 =num2str(0);
rs1 =num2str(0); rs2 =num2str(0); rs3 =num2str(0);rs4 =num2str(0);
qs1 =num2str(0); qs2 =num2str(0); qs3 =num2str(0);qs4 =num2str(0);

% creates a uipanel to display sensor data
hp   = uipanel('parent',hMainfig,...
             'Position',[.0 .75 1 1]);
           
hbsx = uicontrol('Style', 'text','Parent',hp,'String','x', ...
                 'fontSize', 16,...                 
                 'Position',[132 140 100 26]);
pox = getpixelposition(hbsx);             
             
hbsy = uicontrol('Style', 'text','Parent',hp,'String','y', ...
                 'fontSize', 16,...                 
                 'Position',[110 0 0 0] + pox);
poy = getpixelposition(hbsy);  


hbsz = uicontrol('Style', 'text','Parent',hp,'String','z', ...
                 'fontSize', 16,...                 
                 'Position',[100 0 0 0] + poy);
poz = getpixelposition(hbsz);             

hbsa = uicontrol('Style', 'text','Parent',hp,'String','azimuth', ...
                 'fontSize', 16,...                 
                 'Position',[90 0 0 0] + poz);
poa = getpixelposition(hbsa);             

hbse = uicontrol('Style', 'text','Parent',hp,'String','elevation', ...
                 'fontSize', 16,...                 
                 'Position',[100 0 0 0] + poa);
poe = getpixelposition(hbse);        

hbsr = uicontrol('Style', 'text','Parent',hp,'String','roll', ...
                 'fontSize', 16,...                 
                 'Position',[100 0 0 0] + poe);
por = getpixelposition(hbsr);  

hbsq = uicontrol('Style', 'text','Parent',hp,'String','quality', ...
                 'fontSize', 16,...                 
                 'Position',[120 0 0 0] + por);
poq = getpixelposition(hbsq);             

hbs1 = uicontrol('Style', 'text','Parent',hp,'String','sensor 1',...
                 'fontSize', 16,...
                 'Position',[18 110 100 26]);
pos1 = getpixelposition(hbs1);             

hbs2 = uicontrol('Style', 'text','Parent',hp,'String','sensor 2',...
                 'fontSize', 16,...
                 'Position',[18 80 100 26]);
pos2 = getpixelposition(hbs2);             
             

hbs3 = uicontrol('Style', 'text','Parent',hp,'String','sensor 3',...
                 'fontSize', 16,...
                 'Position',[18 50 100 26]);
pos3 = getpixelposition(hbs3);             
                          
hbs4 = uicontrol('Style', 'text','Parent',hp,'String','sensor 4',...
                 'fontSize', 16,...
                 'Position',[18 20 100 26]);
pos4 = getpixelposition(hbs4);             
             
             
             
% x-Position
hbsx1 = uicontrol('Style', 'text','Parent',hp,'String','x1',...
                 'fontSize', 12,...                 
                 'tag', xs1,'Callback', 'cla','Position',[120 0 0 0] + pos1);
posx1 = getpixelposition(hbsx1);

hbsx2 = uicontrol('Style', 'text','Parent',hp,'String','x2',...
                 'fontSize', 12,...                 
                 'tag', xs2,'Callback', 'cla','Position',[120 0 0 0] + pos2);
posx2 = getpixelposition(hbsx2);
             
hbsx3 = uicontrol('Style', 'text','Parent',hp,'String','x3',...
                 'fontSize', 12,...                 
                 'tag', xs3,'Callback', 'cla','Position',[120 0 0 0] + pos3);
posx3 = getpixelposition(hbsx3);
             
hbsx4 = uicontrol('Style', 'text','Parent',hp,'String','x4',...
                 'fontSize', 12,...                 
                 'tag', xs4,'Callback', 'cla','Position',[120 0 0 0] + pos4);
posx4 = getpixelposition(hbsx4);

% y-Position
[hbsy1 hbsy2 hbsy3 hbsy4 posy1 posy2 posy3 posy4] =  ...
positionMoh(hp,xs1,xs2,xs3,xs4,posx1,posx2,posx3,posx4);
% z-Position
[hbsz1 hbsz2 hbsz3 hbsz4 posz1 posz2 posz3 posz4] =  ...
positionMoh(hp,ys1,ys2,ys3,ys4,posy1,posy2,posy3,posy4);
% azimuth
[hbsa1 hbsa2 hbsa3 hbsa4 posa1 posa2 posa3 posa4] =  ...
positionMoh(hp,as1,as2,as3,as4,posz1,posz2,posz3,posz4);
% elevation
[hbse1 hbse2 hbse3 hbse4 pose1 pose2 pose3 pose4] =  ...
positionMoh(hp,es1,es2,es3,es4,posa1,posa2,posa3,posa4);
% roll
[hbsr1 hbsr2 hbsr3 hbsr4 posr1 posr2 posr3 posr4] =  ...
positionMoh(hp,rs1,rs2,rs3,rs4,pose1,pose2,pose3,pose4);
% quality
[hbsq1 hbsq2 hbsq3 hbsq4 posq1 posq2 posq3 posq4] =  ...
positionMoh(hp,rs1,rs2,rs3,rs4,posr1,posr2,posr3,posr4);


% Initialize sensors data to zero
set(hbsx1, 'String', xs1);set(hbsx2, 'String', xs2);set(hbsx3, 'String', xs3);set(hbsx4, 'String', xs4);
set(hbsy1, 'String', ys1);set(hbsy2, 'String', ys2);set(hbsy3, 'String', ys3);set(hbsy4, 'String', ys4);
set(hbsz1, 'String', zs1);set(hbsz2, 'String', zs2);set(hbsz3, 'String', zs3);set(hbsz4, 'String', zs4);
set(hbsa1, 'String', as1);set(hbsa2, 'String', as2);set(hbsa3, 'String', as3);set(hbsa4, 'String', as4);
set(hbse1, 'String', es1);set(hbse2, 'String', es2);set(hbse3, 'String', es3);set(hbse4, 'String', es4);
set(hbsr1, 'String', rs1);set(hbsr2, 'String', rs2);set(hbsr3, 'String', rs3);set(hbsr4, 'String', rs4);
set(hbsq1, 'String', qs1);set(hbsq2, 'String', qs2);set(hbsq3, 'String', qs3);set(hbsq4, 'String', qs4);



% Setup Handlge Graphics for high speed animation
vec_length = 1;  % length of vector on plot showing tracker axes

% Get synchronous Record
sm.x = 0;   sm.y = 0;   sm.z = 0; sm.a = 0;   sm.e = 0;  sm.r = 0;
sm.time = 0;    sm.quality = 0;
Record  = libstruct('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD', sm);  
pRecord = libpointer('tagDOUBLE_POSITION_ANGLES_TIME_Q_RECORD', Record);

% Define an array of colors for each tracker
colorstr_ar = [{'r'} {'b'} {'g'} {'m'} {'y'} {'c'} ];
while length(colorstr_ar)<NumSenPres  
    colorstr_ar = [colorstr_ar colorstr_ar]; % repeat colors if array is not long enough
end
hold on

for count = 1:4
    if SensorNumAttached(count)
            % Get synchronous Record
            Error   = calllib('ATC3DG', 'GetAsynchronousRecord', count - 1, pRecord, 64);
            errorHandler(Error);
            
            Pos(count, 1)   = Record.x;
            Pos(count, 2)   = Record.y;
            Pos(count, 3)   = Record.z;
            Ang(count, 1)   = Record.a;
            Ang(count, 2)   = Record.e;
            Ang(count, 3)   = Record.r;   
            Quality(count)   = Record.quality;  
            
                   
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
    else
            senSorPre(count) = 0;                      
    end
end
indsenSorPre =  find(senSorPre == 1);
 
        
for itracker=1:NumSenPres 
    
    pos = Pos(indsenSorPre(itracker),X:Z);
    ang = Ang(indsenSorPre(itracker),X:Z);
    R = tracker_Rot(ang* 0.017453292519943);            % Calculate Rotation matrix R whose columsn are the unit axes of each tracker coordinate system
    xvec = pos' + R(:,X)*vec_length;  
    yvec = pos' + R(:,Y)*vec_length;
    zvec = pos' + R(:,Z)*vec_length;
    
    colorstr = char(colorstr_ar((itracker)));
    hreal_time_xvec(indsenSorPre(itracker)) = plot3([pos(X);xvec(X)],[pos(Y);xvec(Y)],[pos(Z);xvec(Z)],colorstr,'EraseMode','xor');
    hreal_time_yvec(indsenSorPre(itracker)) = plot3([pos(X);yvec(X)],[pos(Y);yvec(Y)],[pos(Z);yvec(Z)],colorstr,'EraseMode','xor');
    hreal_time_zvec(indsenSorPre(itracker)) = plot3([pos(X);zvec(X)],[pos(Y);zvec(Y)],[pos(Z);zvec(Z)],colorstr,'EraseMode','xor');
    
    
    hreal_time_xtext(indsenSorPre(itracker)) = text(xvec(X),xvec(Y),xvec(Z),['x' int2str(indsenSorPre(itracker))],'Color',colorstr,'EraseMode','xor');
    hreal_time_ytext(indsenSorPre(itracker)) = text(yvec(X),yvec(Y),yvec(Z),['y' int2str(indsenSorPre(itracker))],'Color',colorstr,'EraseMode','xor');
    hreal_time_ztext(indsenSorPre(itracker)) = text(zvec(X),zvec(Y),zvec(Z),['z' int2str(indsenSorPre(itracker))],'Color',colorstr,'EraseMode','xor');
end 

axis vis3d  % set axis for proper 3D visulization
% Free memory
clear Record ;  
clear  pRecord;


