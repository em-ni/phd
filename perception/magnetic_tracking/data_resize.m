function data_resize()
% Resize callback to keep text at upper left corner of window

    global hp;
    global numBoards;
    global xstart;
    global xstep;
    % ystart is not global
    global ystep;
    global height;
    global width;
    global sensorwidth; % width of sensor label column

    set(hp, 'Units', 'pixels');
    p = get(hp,'Position');
    set(hp, 'Units', 'normalized');
    ystart = p(2)+p(4)-40;

    children = get(hp, 'Children');
    numchildren = length(children);
    %children are numbered in last to first
    %headers
    %hbss
    pos = [xstart,ystart,width,height];
    set(children(numchildren), 'Position', pos);
    %hbsx
    pos = [xstart+sensorwidth,ystart,width,height];
    set(children(numchildren-1), 'Position', pos);
    %hbsy
    pos = pos + [xstep,0,0,0];
    set(children(numchildren-2), 'Position', pos);
    %hbsz
    pos = pos + [xstep,0,0,0];
    set(children(numchildren-3), 'Position', pos);
    %hbsa
    pos = pos + [xstep,0,0,0];
    set(children(numchildren-4), 'Position', pos);
    %hbse
    pos = pos + [xstep,0,0,0];
    set(children(numchildren-5), 'Position', pos);
    %hbsr
    pos = pos + [xstep,0,0,0];
    set(children(numchildren-6), 'Position', pos);
    %hbsq
    pos = pos + [xstep,0,0,0];
    set(children(numchildren-7), 'Position', pos);
   
    %counter to index through children array
    ch = numchildren-8;
    %sensor numbers
    pos = [xstart, ystart-ystep, sensorwidth, height];
    for kk = 1: 4 * numBoards
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end
    
    %data
    % x-position
    pos = [xstart+sensorwidth, ystart-ystep, width, height];
    for kk = 1: 4 * numBoards ,
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end

    % y-position
    pos = [xstart+sensorwidth+xstep, ystart-ystep, width, height];
    for kk = 1: 4 * numBoards ,
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end
    
    % z-position
    pos = [xstart+sensorwidth+2*xstep, ystart-ystep, width, height];
    for kk = 1: 4 * numBoards ,
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end

    % azimuth data
    pos = [xstart+sensorwidth+3*xstep, ystart-ystep, width, height];
    for kk = 1: 4 * numBoards ,
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end
    
    % elevation data
    pos = [xstart+sensorwidth+4*xstep, ystart-ystep, width, height];
    for kk = 1: 4 * numBoards ,
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end
    
    % roll data
    pos = [xstart+sensorwidth+5*xstep, ystart-ystep, width, height];
    for kk = 1: 4 * numBoards ,
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end
    
    % quality data
    pos = [xstart+sensorwidth+6*xstep, ystart-ystep, width, height];
    for kk = 1: 4 * numBoards ,
        set(children(ch), 'Position', pos);
        pos = pos + [0, -ystep, 0, 0];
        ch = ch - 1;
    end
end

