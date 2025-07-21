function tracker_close

% tracker_close: Free memory, turns-off the transmitter and unload the
% library
% Ascension Technology Corporation 

global libstring;

% Free memory
clear Record1;
clear pRecord1;
clear Record2;
clear pRecord2;
clear Record3;
clear pRecord3;
clear Record4;
clear pRecord4;

clear Record;
clear pRecord;

clear RecordA;
clear pRecordA;
clear Record1;
clear pRecord1;

clear Records0;
clear pRecords0;
clear Records1;
clear pRecords1;
clear Records2;
clear pRecords2;
clear Records3;
clear pRecords3;

% Turn off Transmitter
Error     = calllib(libstring, 'SetSystemParameter', 0, -1, 2);
errorHandler(Error);

% Close tracker 
Error  = calllib(libstring, 'CloseBIRDSystem');
errorHandler(Error);


% Unload the library
if libisloaded(libstring)
%     unloadlibrary ATC3DG;
end
