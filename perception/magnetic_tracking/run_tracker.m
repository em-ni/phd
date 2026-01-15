% Run this program to start tracker menu
% See "Matlab Tracker User Guide Rev 1.0.doc" file for documentation
% Ascension Technology Corporation 

clear variables;
close all;
clc;
warning('off');

% Define global variables used in gui
global gui  % paramters passed to gui (like start_proc, and stop_proc)
global hMainfig;

fprintf('Click "Setup Tracker" on the "Run Tracker" menu to initialize the system \n\n');
RunDribayFirst = 0;
ON=1; OFF=0;


% Setup Menu 
header_str = 'Run Tracker';
clear menu_str;
im=1;
menu_str(im) = {'Setup Tracker'}; im = im+1;
menu_str(im) = {'Read Position of Attached Sensor and Display at the MATLAB Command Window'}; im = im+1;
menu_str(im) = {'Stream Data on Tracker then Plot'}; im = im+1;
menu_str(im) = {'Save Data Stream to File'}; im = im+1;
menu_str(im) = {'Tracker Real-Time Data Animation'}; im = im+1;
menu_str(im) = {'Tracker Real-Time Graphic Animation'}; im = im+1;
menu_str(im) = {'Ian Live Tracker Real-Time Graphic Animation'}; im = im+1;
menu_str(im) = {'Clear Figure'}; im = im+1;
menu_str(im) = {'Close Tracker and Unload Library'}; im = im+1;
menu_str(im) = {'Exit'}; im = im+1;



% setup figure
set(0,'Units','pixels') 
scnsize  = get(0,'ScreenSize');
pos2     = [ scnsize(3)/2-30, scnsize(4)/2-30 ,scnsize(3)/2 - 70, scnsize(4)/2- 70 ];
hMainfig = figure('Position',pos2);
%axes('position',[.1  .1  .8  .6])
view(3)
axis vis3d  % set axis for proper 3D visulization
setup_views    % load pull downmenu for pre defined views


in_loop_run_laryn = ON;
% loop until 'Exit' is selected
while in_loop_run_laryn       
    im = menu(header_str,menu_str);  % call menu and return selection
    
    switch char(menu_str(im))
        case 'Setup Tracker'
           RunDribayFirst = 1;
           tracker_setup;  % setup tracker        

        case 'Read Position of Attached Sensor and Display at the MATLAB Command Window'
            tracker_position_angles_Quality;% display data to the MATLAB command window

        case  'Stream Data on Tracker then Plot'
            tracker_stream_plot(0);% Plot data stream

        case 'Save Data Stream to File'
             save_Stream_Data;% save data to files

        case 'Tracker Real-Time Data Animation'
             real_time_Animation_Text; % data display
             
        case 'Tracker Real-Time Graphic Animation'
             real_time_Animation_Graphic; % graphical display     
             
        case 'Ian Live Tracker Real-Time Graphic Animation'
             tracker_live_plot(2); % Ian's state-of-the-art display
            xlim([0 20]);
            ylim([-10 10]);
            zlim([-10 10]);
        case 'Close Tracker and Unload Library' 
             tracker_close; % close tracker
             
        case 'Clear Figure'
           h_ar = get(gca,'Children');   % find all graphics handles and
           drawnow;                      % delete them 
           delete(h_ar);                  

        case 'Exit'
           in_loop_run_laryn = OFF;
           close all;

        otherwise
            error('Invalid menu option in run_bird');
    end
end



