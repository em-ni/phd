function save_Stream_Data
% save_Stream_Data: writes attached sensors position, angles and
% quality number to a file
% Ascension Technology Corporation 


global SensorNumAttached; % number of attached sensors
global hMainfig;
global numBoards;% number of boards

clf;

NumSenPres   = sum(SensorNumAttached);
indsenSorPre =  find(SensorNumAttached == 1);

[Pos, Ang, TimStam, Quality, Error] = tracker_stream_save(0);

fprintf('\n\n');
fprintf('Sensors data is saved in the current MATLAB directory.\n');
fprintf('For example: Data_Sensor_3.dat, where 3 is the sensor number\n\n');

for count=1:NumSenPres
    PosTemp = Pos(: , :,indsenSorPre(count));
    AngTemp = Ang(: , :,indsenSorPre(count));
    QuaTemp = Quality(: , :,indsenSorPre(count));
   
    % TimStam   =  TimStam  -  TimStam(1);
    temp      =  [PosTemp AngTemp TimStam' QuaTemp];
    ind       =  int2str(indsenSorPre(count));
    fileName  = char(strcat({'Data_Sensor_'},{ind},{ '.mat'}));
    fid       = fopen(fileName, 'wt');
    fprintf(fid,'Sensor %g position, angle , timestamp, and quality in inches, degrees and seconds, respectively\n\n',...
            indsenSorPre(count));
    fprintf(fid,' x            y            z            a            e            r            timestamp          quality\n');             
    fprintf(fid,'----------------------------------------------------------------------------------------------------------------\n\n');

    fprintf(fid, '%+3.8f %+3.8f %+3.8f %+3.8f %+3.8f %+3.8f %+3.8f %+3g\n', temp');
    fclose(fid);
    save(fileName,'temp');
end