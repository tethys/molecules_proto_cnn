%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = LSAGDR (fx, gradf, parameter)
% Purpose:   Implementation of AGD with line search and adaptive restart.     
% Parameter: x0         - Initial estimate.
%            maxit      - Maximum number of iterations.
%            tolx       - Error toleration for stopping condition.
%            Lips       - Lipschitz constant for gradient.
%            strcnvx    - Strong convexity parameter of f(x).
%*************************** LIONS@EPFL ***********************************
function [x, info] = LSAGDR(fx, gradf, parameter)
        
    fprintf('%s\n', repmat('*', 1, 68));

    % Set the clock.
    time1       = tic;
    timestart   = toc(time1);

    % Initialize x, y, t and fx(x).
    x_next       = parameter.x0;
    y            = parameter.x0;
    t_next       = 1;
    fval         = fx(parameter.x0); 

    % Main loop.
    for iter = 1:parameter.maxit
              
        x           = x_next;
        t           = t_next;

        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)  = toc(time1) - timestart;
        info.fx(iter, 1)        = fx(x);

        % Print the information.
        fprintf('Iter = %4d, f(x) = %5.3e\n', ...
                iter, info.fx(iter, 1));
            
        % Start the clock.
        timestart   = toc(time1);
        
        % Evaluate the gradient vectors.
        d = -gradf(x);
        % Approximate local Lipschitz constant.
        kappa = 0.1;
        for j=1:100
            if fx(x + 2^j/L*d) > fx(x) + kappa* 2^j/L* gradf(x)'*d;
                break;
            end
        end
        alpha = 2^(j-1)/L;
        % Update the next iteration.
        x_next = y - 1/parameter.Lips * gradf(y);
        % Restart the iteration if necessary.
        
        t_next = 0.5* (1 + sqrt(1+ 4* t*t));
        y = x_next + (t - 1)/t_next*(x_next - x); 
        
        % Check stopping criterion.
        if norm(x_next - x) <= parameter.tolx 
            break;
        end
        
    end

    % Finalization.
    info.iter           = iter;
    info.time           = cumsum(info.itertime);
    info.totaltime      = info.time(iter);
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
