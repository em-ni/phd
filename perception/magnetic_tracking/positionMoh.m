function [hbsa1 hbsa2 hbsa3 hbsa4 posa1 posa2 posa3 posa4] =  positionMoh(hp,as1,as2,as3,as4,posz1,posz2,posz3,posz4) 
% creates 4 uicontrol in the object specified by the handle hp.
% input: hp: parent object
%        as1, as2, as3 and as4: User-specified object label initialization
%        posz1,posz2,posz3 and posz4: position vectors used to compute the
%                                     size and location of the 4 uicontrols
%output: hbsa1, hbsa2, hbsa3 and hbsa4: the 4 unicontrol objects
%      posa1, posa2, posa3 and posa4:size and location of the 4 uicontrols 

% Ascension Technology Corporation 


% first uicontrol
hbsa1 = uicontrol('Style', 'text','Parent',hp,'String','z1',...
                 'fontSize', 12,...                 
                 'tag', as1,'Callback', 'cla','Position',[100 0 0 0] + posz1);
% get size and location             
posa1 = getpixelposition(hbsa1);

hbsa2 = uicontrol('Style', 'text','Parent',hp,'String','z2',...
                 'fontSize', 12,...                 
                 'tag', as2,'Callback', 'cla','Position',[100 0 0 0] + posz2);
% get size and location             
posa2 = getpixelposition(hbsa2);
             
hbsa3 = uicontrol('Style', 'text','Parent',hp,'String','z3',...
                 'fontSize', 12,...                 
                 'tag', as3,'Callback', 'cla','Position',[100 0 0 0] + posz3);
% get size and location             
posa3 = getpixelposition(hbsa3);
             
hbsa4 = uicontrol('Style', 'text','Parent',hp,'String','z4',...
                 'fontSize', 12,...                 
                 'tag', as4,'Callback', 'cla','Position',[100 0 0 0] + posz4);
% get size and location             
posa4 = getpixelposition(hbsa4);


