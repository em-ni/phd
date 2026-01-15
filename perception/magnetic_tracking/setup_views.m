% setup_views: create a new menu on the menu bar at the top of the figure.
% This enable the user to specify the viewpoint to two or three dimensional
% view
% Ascension Technology Corporation 

% Create a new menu on the menu bar at the top of the figure window with the handle hfig
h_menu = uimenu(gcf,'Label','Defined Views');

% submenus are added by specfiying the handle to the column menu
uimenu(h_menu,'Label','Top View', ...
                'Callback', ' view(90,-90) ');
uimenu(h_menu,'Label','Side View', ...
                'Callback', 'view(90,0)');
uimenu(h_menu,'Label','Front View', ...
                'Callback', 'view(180,0)');
uimenu(h_menu,'Label','Isometric View', ...
                'Callback', 'view(-37.5,30)');
