%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = QNM (fx, gradf, hessf, parameter)
% Purpose:   Implementation of the quasi-Newton method with BFGS update.     
% Parameter: x0         - Initial estimate.
%            maxit      - Maximum number of iterations.
%            tolx       - Error toleration for stopping condition.
%            Lips       - Lipschitz constant for gradient.
%            strcnvx    - Strong convexity parameter of f(x).
%*************************** LIONS@EPFL ***********************************
function [x, info] = QNM(fx, gradf, hessf, parameter)
    
    fprintf('%s\n', repmat('*', 1, 68));

    % Set the clock.
    time1       = tic;
    timestart   = toc(time1);
    
    % Initialize x, y and t.
    x_next          = parameter.x0;
    
    % Main loop.
    for iter = 1:parameter.maxit
        
        x           = x_next;
        
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)  = toc(time1) - timestart;
        info.fx(iter, 1)        = fx(x);
                
        % Print the information.
        fprintf('Iter = %4d, f(x) = %5.3e\n', ...
                iter, info.fx(iter, 1));
                    
        % Start the clock.
        timestart   = toc(time1);
        
        % Compute the search direction d (you are asked to implement this).
        '???' 
        
        % Compute the step size alpha by a backtracking linesearch.
        % (you are asked to implement this)
        '???'
        
        % Update the next iteration.
        x_next = x + alpha*dx;

        % Update matrix Bk by BFGS.
        '???'
        
        % Check stopping criterion.
        if norm(alpha*dx) <= parameter.tolx 
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
