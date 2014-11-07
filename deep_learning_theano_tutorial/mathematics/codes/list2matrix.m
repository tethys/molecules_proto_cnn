%*******************  EE556 - Mathematics of Data  ************************
% Function:  [ A ] = list2matrix( filename )
% Purpose:   Converts downloaded datasets to mat.     
% Input:     filename - String consisting the path to the file.
%                       e.g. 'datasets/Wiki-Vote.txt'
%*************************** LIONS@EPFL ***********************************
function [ A ] = list2matrix( filename )

A = dlmread(filename,'\t',4, 0);
if min(A) == 0
    A = A + 1;
end
if min(min(A)) == 0
    A = A+1;
end

A = sparse(A(:,1),A(:,2),1,max(max(A)),max(max(A)));

end

