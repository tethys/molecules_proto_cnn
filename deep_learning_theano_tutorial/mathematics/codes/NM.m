%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, output] = NM (fx, gradf, hessf, parameter)
% Purpose:   Implementation of the Newton method.     
% Parameter: x0         - Initial estimate.
%            maxit      - Maximum number of iterations.
%            tolx       - Error toleration for stopping condition.
%            Lips       - Lipschitz constant for gradient.
%            strcnvx    - Strong convexity parameter of f(x).
%*************************** LIONS@EPFL ***********************************
function [x, output] = NM(fx, gradf, hessf, parameter)
    
    fprintf('%s\n', repmat('*', 1, 68));
    
    % Set the clock.
    time1       = tic;
    timestart   = toc(time1);
    
    % Initialize x.
    x_next     = parameter.x0;
    
    % Main loop.
    for iter = 1:parameter.maxit
        
        x       = x_next;
        
        % Compute error and save data to be plotted later on.
        output.itertime(iter ,1)  = toc(time1) - timestart;
        output.fx(iter, 1)        = fx(x);
                
        % Print the information.
        fprintf('Iter = %4d, f(x) = %5.3e\n', ...
                iter, output.fx(iter, 1));
                    
        % Start the clock.
        timestart   = toc(time1);
        
        % Compute the search direction d (you are asked to implement this).
        '???' 
        
        % Compute the step size alpha by a backtracking linesearch.
        % (you are asked to implement this)
        '???'
        
        % Update the next iteration.
        x_next = x + alpha*dx;
        
        % Check stopping criterion.
        if norm(alpha*dx) <= parameter.tolx 
            break;
        end
        
    end 

    % Finalization.
    output.iter           = iter;
    output.time           = cumsum(output.itertime);
    output.totaltime      = output.time(iter);
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************