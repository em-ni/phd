function tracker_setup;

% tracker_setup: sets up tracker system for data acquisition.  This include: 
% load library, initialize the tracker, etc..
% Ascension Technology Corporation 


global libstring;
global SensorNumAttached; % number of attached sensors
global measurementRate; % measurement rate
global numBoards;
disp('Initialization in progress; please wait a moment...')

% Free memory
clear Record;
clear pRecord;

% Determine if platform is 32 or 64 bit
if (strcmp(computer, 'PCWIN64'))
    libstring = 'ATC3DG64';
else
    libstring = 'ATC3DG';
end

% Check if library is loaded
if libisloaded(libstring)
%     unloadlibrary ATC3DG;
end

% Loading the ATC3DG library
[notfound,warnings]=loadlibrary(libstring, 'ATC3DG.h');

% Check if loading library is successful
CheckLibrary = libisloaded(libstring);
if not(CheckLibrary)
    error('Problem load library: loadlibrary');
    clear all;
end

% Initialize the system
temp  = calllib(libstring, 'InitializeBIRDSystem');
errorHandler(temp);
% Get system configuration
Record3         = libstruct('tagSYSTEM_CONFIGURATION'); 
Record3.agcMode = 0;
pRecord3        = libpointer('tagSYSTEM_CONFIGURATION', Record3);
temp3           = calllib(libstring, 'GetBIRDSystemConfiguration', pRecord3);
errorHandler(temp3)
measurementRate = Record3.measurementRate;

numBoards       = Record3.numberBoards;

% Get sensor configuration
if numBoards == 1
    Records0        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records0.type   = 11;
    pRecords0       = libpointer('tagSENSOR_CONFIGURATION', Records0);
    temps0          = calllib(libstring, 'GetSensorConfiguration', 0, pRecords0);
    errorHandler(temps0);

    Records1        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records1.type   = 11;
    pRecords1       = libpointer('tagSENSOR_CONFIGURATION', Records1);
    temps1          = calllib(libstring, 'GetSensorConfiguration', 1, pRecords1);
    errorHandler(temps1);

    Records2        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records2.type   = 11;
    pRecords2       = libpointer('tagSENSOR_CONFIGURATION', Records2);
    temps2          = calllib(libstring, 'GetSensorConfiguration', 2, pRecords2);
    errorHandler(temps2);

    Records3        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records3.type   = 11;
    pRecords3       = libpointer('tagSENSOR_CONFIGURATION', Records3);
    temps3          = calllib(libstring, 'GetSensorConfiguration', 3, pRecords3);
    errorHandler(temps3);
    
    SensorNumAttached  = [Records0.attached Records1.attached Records2.attached ...
                      Records3.attached ];
    
elseif numBoards == 2
    Records0        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records0.type   = 11;
    pRecords0       = libpointer('tagSENSOR_CONFIGURATION', Records0);
    temps0          = calllib(libstring, 'GetSensorConfiguration', 0, pRecords0);
    errorHandler(temps0);

    Records1        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records1.type   = 11;
    pRecords1       = libpointer('tagSENSOR_CONFIGURATION', Records1);
    temps1          = calllib(libstring, 'GetSensorConfiguration', 1, pRecords1);
    errorHandler(temps1);

    Records2        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records2.type   = 11;
    pRecords2       = libpointer('tagSENSOR_CONFIGURATION', Records2);
    temps2          = calllib(libstring, 'GetSensorConfiguration', 2, pRecords2);
    errorHandler(temps2);

    Records3        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records3.type   = 11;
    pRecords3       = libpointer('tagSENSOR_CONFIGURATION', Records3);
    temps3          = calllib(libstring, 'GetSensorConfiguration', 3, pRecords3);
    errorHandler(temps3);

    Records4        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records4.type   = 11;
    pRecords4       = libpointer('tagSENSOR_CONFIGURATION', Records4);
    temps4          = calllib(libstring, 'GetSensorConfiguration', 4, pRecords4);
    errorHandler(temps4);

    Records5        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records5.type   = 11;
    pRecords5       = libpointer('tagSENSOR_CONFIGURATION', Records5);
    temps5          = calllib(libstring, 'GetSensorConfiguration', 5, pRecords5);
    errorHandler(temps5);

    Records6        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records6.type   = 11;
    pRecords6       = libpointer('tagSENSOR_CONFIGURATION', Records6);
    temps6          = calllib(libstring, 'GetSensorConfiguration', 6, pRecords6);
    errorHandler(temps6);

    Records7        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records7.type   = 11;
    pRecords7       = libpointer('tagSENSOR_CONFIGURATION', Records7);
    temps7          = calllib(libstring, 'GetSensorConfiguration', 7, pRecords7);
    errorHandler(temps7);
    
    SensorNumAttached  = [Records0.attached Records1.attached Records2.attached ...
                      Records3.attached Records4.attached Records5.attached ...
                      Records6.attached Records7.attached ];    
elseif numBoards == 3
    Records0        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records0.type   = 11;
    pRecords0       = libpointer('tagSENSOR_CONFIGURATION', Records0);
    temps0          = calllib(libstring, 'GetSensorConfiguration', 0, pRecords0);
    errorHandler(temps0);

    Records1        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records1.type   = 11;
    pRecords1       = libpointer('tagSENSOR_CONFIGURATION', Records1);
    temps1          = calllib(libstring, 'GetSensorConfiguration', 1, pRecords1);
    errorHandler(temps1);

    Records2        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records2.type   = 11;
    pRecords2       = libpointer('tagSENSOR_CONFIGURATION', Records2);
    temps2          = calllib(libstring, 'GetSensorConfiguration', 2, pRecords2);
    errorHandler(temps2);

    Records3        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records3.type   = 11;
    pRecords3       = libpointer('tagSENSOR_CONFIGURATION', Records3);
    temps3          = calllib(libstring, 'GetSensorConfiguration', 3, pRecords3);
    errorHandler(temps3);

    Records4        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records4.type   = 11;
    pRecords4       = libpointer('tagSENSOR_CONFIGURATION', Records4);
    temps4          = calllib(libstring, 'GetSensorConfiguration', 4, pRecords4);
    errorHandler(temps4);

    Records5        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records5.type   = 11;
    pRecords5       = libpointer('tagSENSOR_CONFIGURATION', Records5);
    temps5          = calllib(libstring, 'GetSensorConfiguration', 5, pRecords5);
    errorHandler(temps5);

    Records6        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records6.type   = 11;
    pRecords6       = libpointer('tagSENSOR_CONFIGURATION', Records6);
    temps6          = calllib(libstring, 'GetSensorConfiguration', 6, pRecords6);
    errorHandler(temps6);

    Records7        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records7.type   = 11;
    pRecords7       = libpointer('tagSENSOR_CONFIGURATION', Records7);
    temps7          = calllib(libstring, 'GetSensorConfiguration', 7, pRecords7);
    errorHandler(temps7);

    Records8        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records8.type   = 11;
    pRecords8       = libpointer('tagSENSOR_CONFIGURATION', Records8);
    temps8          = calllib(libstring, 'GetSensorConfiguration', 8, pRecords8);
    errorHandler(temps8);

    Records9        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records9.type   = 11;
    pRecords9       = libpointer('tagSENSOR_CONFIGURATION', Records9);
    temps9          = calllib(libstring, 'GetSensorConfiguration', 9, pRecords9);
    errorHandler(temps9);

    Records10        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records10.type   = 11;
    pRecords10       = libpointer('tagSENSOR_CONFIGURATION', Records10);
    temps10          = calllib(libstring, 'GetSensorConfiguration', 10, pRecords10);
    errorHandler(temps10);

    Records11        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records11.type   = 11;
    pRecords11       = libpointer('tagSENSOR_CONFIGURATION', Records11);
    temps11          = calllib(libstring, 'GetSensorConfiguration', 11, pRecords11);
    errorHandler(temps11);
    SensorNumAttached  = [Records0.attached Records1.attached Records2.attached ...
                      Records3.attached Records4.attached Records5.attached ...
                      Records6.attached Records7.attached Records8.attached ...
                      Records9.attached Records10.attached Records11.attached];
else
    Records0        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records0.type   = 11;
    pRecords0       = libpointer('tagSENSOR_CONFIGURATION', Records0);
    temps0          = calllib(libstring, 'GetSensorConfiguration', 0, pRecords0);
    errorHandler(temps0);

    Records1        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records1.type   = 11;
    pRecords1       = libpointer('tagSENSOR_CONFIGURATION', Records1);
    temps1          = calllib(libstring, 'GetSensorConfiguration', 1, pRecords1);
    errorHandler(temps1);

    Records2        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records2.type   = 11;
    pRecords2       = libpointer('tagSENSOR_CONFIGURATION', Records2);
    temps2          = calllib(libstring, 'GetSensorConfiguration', 2, pRecords2);
    errorHandler(temps2);

    Records3        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records3.type   = 11;
    pRecords3       = libpointer('tagSENSOR_CONFIGURATION', Records3);
    temps3          = calllib(libstring, 'GetSensorConfiguration', 3, pRecords3);
    errorHandler(temps3);

    Records4        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records4.type   = 11;
    pRecords4       = libpointer('tagSENSOR_CONFIGURATION', Records4);
    temps4          = calllib(libstring, 'GetSensorConfiguration', 4, pRecords4);
    errorHandler(temps4);

    Records5        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records5.type   = 11;
    pRecords5       = libpointer('tagSENSOR_CONFIGURATION', Records5);
    temps5          = calllib(libstring, 'GetSensorConfiguration', 5, pRecords5);
    errorHandler(temps5);

    Records6        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records6.type   = 11;
    pRecords6       = libpointer('tagSENSOR_CONFIGURATION', Records6);
    temps6          = calllib(libstring, 'GetSensorConfiguration', 6, pRecords6);
    errorHandler(temps6);

    Records7        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records7.type   = 11;
    pRecords7       = libpointer('tagSENSOR_CONFIGURATION', Records7);
    temps7          = calllib(libstring, 'GetSensorConfiguration', 7, pRecords7);
    errorHandler(temps7);

    Records8        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records8.type   = 11;
    pRecords8       = libpointer('tagSENSOR_CONFIGURATION', Records8);
    temps8          = calllib(libstring, 'GetSensorConfiguration', 8, pRecords8);
    errorHandler(temps8);

    Records9        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records9.type   = 11;
    pRecords9       = libpointer('tagSENSOR_CONFIGURATION', Records9);
    temps9          = calllib(libstring, 'GetSensorConfiguration', 9, pRecords9);
    errorHandler(temps9);

    Records10        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records10.type   = 11;
    pRecords10       = libpointer('tagSENSOR_CONFIGURATION', Records10);
    temps10          = calllib(libstring, 'GetSensorConfiguration', 10, pRecords10);
    errorHandler(temps10);

    Records11        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records11.type   = 11;
    pRecords11       = libpointer('tagSENSOR_CONFIGURATION', Records11);
    temps11          = calllib(libstring, 'GetSensorConfiguration', 11, pRecords11);
    errorHandler(temps11);
    
    Records12        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records12.type   = 11;
    pRecords12       = libpointer('tagSENSOR_CONFIGURATION', Records12);
    temps12          = calllib(libstring, 'GetSensorConfiguration', 12, pRecords12);
    errorHandler(temps12);
    
    Records13        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records13.type   = 11;
    pRecords13       = libpointer('tagSENSOR_CONFIGURATION', Records13);
    temps13          = calllib(libstring, 'GetSensorConfiguration', 13, pRecords13);
    errorHandler(temps13);    
    
    Records14        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records14.type   = 11;
    pRecords14       = libpointer('tagSENSOR_CONFIGURATION', Records14);
    temps14          = calllib(libstring, 'GetSensorConfiguration', 14, pRecords14);
    errorHandler(temps14);

    Records15        = libstruct('tagSENSOR_CONFIGURATION'); 
    Records15.type   = 11;
    pRecords15       = libpointer('tagSENSOR_CONFIGURATION', Records15);
    temps15          = calllib(libstring, 'GetSensorConfiguration', 15, pRecords15);
    errorHandler(temps15);
    
    SensorNumAttached  = [Records0.attached Records1.attached Records2.attached ...
                      Records3.attached Records4.attached Records5.attached ...
                      Records6.attached Records7.attached Records8.attached ...
                      Records9.attached Records10.attached Records11.attached ...
                      Records12.attached Records13.attached Records14.attached ...
                      Records15.attached];
end   

% Turn ON Transmitter
tempSetP    = calllib(libstring, 'SetSystemParameter', 0, 0, 2);
errorHandler(tempSetP);

% Set sensor parameters
var       = int32(26);%(19);

for count = 0:4*numBoards - 1
    Error1 = calllib(libstring, 'SetSensorParameter', count, 0, var, 4);
    errorHandler(Error1);
end


if (temp == 0)
    disp('System initialized: Done')
else
    error('Problem initialising the system: InitializeBIRDSystem');
end


% Free memory
clear Record3;
clear pRecord3;
clear Records0;
clear pRecords0;
clear Records1;
clear pRecords1;
clear Records2;
clear pRecords2;
clear Records3;
clear pRecords3;
