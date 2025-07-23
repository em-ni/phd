function [R] = tracker_Rot(th)
% tracker_Rot: converts three angles from the position sensor to a rotation
% matrix
% Input: th => 3x1 vector of Euler angles as output by bird in Radians
% Output: R => 3x3 rotation matrix.
% Ascension Technology Corporation 


thx = th(3);  % Roll
thy = th(2);  % Elevation
thz = th(1);  % Azimuth


  R(1,1) = cos(thy)*cos(thz);
  R(1,2) = cos(thy)*sin(thz);
  R(1,3) = -sin(thy);
  R(2,1) = -cos(thx)*sin(thz) + sin(thx)*sin(thy)*cos(thz);
  R(2,2) = cos(thx)*cos(thz) + sin(thx)*sin(thy)*sin(thz);
  R(2,3) = sin(thx)*cos(thy);
  R(3,1) = sin(thx)*sin(thz) + cos(thx)*sin(thy)*cos(thz);
  R(3,2) = -sin(thx)*cos(thz) + cos(thx)*sin(thy)*sin(thz);
  R(3,3) = cos(thx)*cos(thy);
  R = R';   % convert to standard notation used in Asada


