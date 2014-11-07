%**************************************************************************
%*************************** LIONS@EPFL ***********************************
%**************************************************************************
clear all
datasetname = 'Wiki-Vote.txt';  % Change ??? appropriately to load data.
    % 'Wiki-Vote.txt'
    %
datasetname = ['data/', datasetname];
A  = list2matrix(datasetname);   % A is adjacency matrix
n  = size(A,1);                  % n is number of nodes
E  = '???';                      % E is transition matrix
p  = 0.15;                       % Damping factor.
Mx = @(x)( '???' );              % An efficient way of implementing M*x 
                                 % where M is PageRank matrix

% Evaluate the Lipschitz constant and strong convexity parameter.
penaltyparameter            = 1;               % You can vary penalty parameter
parameter.Lips              = '???';
parameter.strcnvx           = 0;

% Set parameters and solve numerically with GD, AGD, AGDR, LSGD, LSAGD, LSAGDR.
fprintf(strcat('Numerical solution process is started: \n'));
fx                          = @(x)( '???' );
gradf                       = @(x)( '???' );
parameter.x0                = zeros(n, 1);
parameter.tolx              = 1e-10;            % You can vary tolx and maxit     
parameter.maxit             = 1e5;              % to achieve the convergence. 

[x.GD     , info.GD     ]   = GD     (fx, gradf, parameter);
[x.AGD    , info.AGD    ]   = AGD    (fx, gradf, parameter);
[x.AGDR   , info.AGDR   ]   = AGDR   (fx, gradf, parameter);
[x.LSGD   , info.LSGD   ]   = LSGD   (fx, gradf, parameter);
[x.LSAGD  , info.LSAGD  ]   = LSAGD  (fx, gradf, parameter);
[x.LSAGDR , info.LSAGDR ]   = LSAGDR (fx, gradf, parameter);

% Solve numerically with CG.
sigma                       = 1;                % You can vary regularization parameter
Phix                        = @(x)( '???' );    % Implements Phi_sigma * x for CG method
y                           = '???';            % vector y for CG algorithm
[x.CG     , info.CG     ]   = CG     (fx, Phix, y, parameter);

% Solve numerically with PageRank algorithm (Power method).
parameter.x0                = 1/n*ones(n, 1);
[x.PR     , info.PR     ]   = PR( fx, Mx, parameter );

fprintf(strcat('Numerical solution process is completed. \n'));

plotresults;

%*******************************%
%  EPFL STI IEL LIONS           %
%  1015 LAUSANNE                %
%*******************************%
