%*******************  EE556 - Mathematics of Data  ************************
% Function:  [ x_out, y_out ] = putmarker( x, y, d )       
% Purpose:   Helps to get periodic markers in plots.     
% Inputs:    y -    vertical positions of data points
%            x -    horizontal positions of data points
%            d -    number of markers
%*************************** LIONS@EPFL ***********************************
function [ x_out, y_out ] = putmarker( x, y, d )

markerpoints = (1:2:(2*d-1)) * floor(length(y)/2/d);
if isempty(x)
    x_out = 1:length(y);
    x_out = x_out(markerpoints);
else
    x_out = x(markerpoints);
end
y_out = y(markerpoints);

end

%*******************************%
%  EPFL STI IEL LIONS           %
%  1015 LAUSANNE                %
%*******************************%

