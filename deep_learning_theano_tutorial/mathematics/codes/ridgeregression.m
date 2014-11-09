%**************************************************************************
%*************************** LIONS@EPFL ***********************************
%**************************************************************************
clear all

% Parameters for synthetic data.
cfg.n                       = 1e2;      % number of features
cfg.p                       = 1e3;      % number of dimensions
cfg.noisestd                = 1e-6;     % standard deviation of additive iid gaussian noise (0 for noiseless)
cfg.strcnvx                 = false;    % false = not strongly convex
                                        % true  = strongly convex with, lambda = 0.01*norm(A'*A)

% Methods to be checked.
chk.GD                      = true;    % You can check one or more methods at once.
chk.AGD                     = true;    
chk.AGDR                    = true;    
chk.LSGD                    = true;    
chk.LSAGD                   = false;     
chk.LSAGDR                  = false;    
chk.CG                      = false;    
               
% Generate synthetic data.
A                           = rand(cfg.n, cfg.p);
% Generate s-sparse vector.
xtrue                       = randn(cfg.p, 1);
% Take (noisy) samples.
noise                       = cfg.noisestd*randn(cfg.n, 1);
b                           = A*xtrue + noise;

% Strongly convex OR Convex?
if cfg.strcnvx
  cfg.lambda                = 0.01*norm(A'*A);
else
  cfg.lambda                = 0;
end

% Evaluate the Lipschitz constant and strong convexity parameter.
parameter.Lips              = norm(A'*A + cfg.lambda*eye(cfg.p),2);
parameter.mu                = cfg.lambda;

% Set parameters and solve numerically.
fprintf(strcat('Numerical solution process is started: \n'));
fx                          = @(x)( 0.5*norm(A*x - b)^2 + 0.5*cfg.lambda*norm(x,2)^2 );
gradf                       = @(x)( A'*(A*x - b) + cfg.lambda*x );
parameter.x0                = zeros(cfg.p, 1);
parameter.tolx              = 1e-10;            % You can vary tolx and maxit     
parameter.maxit             = 1e5;              % to achieve the convergence. 


if chk.GD
[x.GD     , info.GD     ]   = GD     (fx, gradf, parameter); end
if chk.AGD 
[x.AGD    , info.AGD    ]   = AGD    (fx, gradf, parameter); end
if chk.AGDR
[x.AGDR   , info.AGDR   ]   = AGDR   (fx, gradf, parameter); end
if chk.LSGD
[x.LSGD   , info.LSGD   ]   = LSGD   (fx, gradf, parameter); end
if chk.LSAGD
[x.LSAGD  , info.LSAGD  ]   = LSAGD  (fx, gradf, parameter); end
if chk.LSAGDR
[x.LSAGDR , info.LSAGDR ]   = LSAGDR (fx, gradf, parameter); end
if chk.CG
[x.CG     , info.CG     ]   = CG     (fx, gradf, parameter, xtrue); end

fprintf(strcat('Numerical solution process is completed. \n'));

% Find x^* and f^* if noisy to plot data.
if cfg.noisestd ~= 0
    xmin                     = pinv(A'*A + cfg.lambda*eye(cfg.p))*A'*b;
    fmin                     = fx(xmin);
end

% Plot the results.
plotresults;



%*******************************%
%  EPFL STI IEL LIONS           %
%  1015 LAUSANNE                %
%*******************************%
