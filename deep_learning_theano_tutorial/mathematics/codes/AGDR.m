%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = AGDR (fx, gradf, parameter)
% Purpose:   Implementation of the AGD with adaptive restart.     
% Parameter: x0         - Initial estimate.
%            maxit      - Maximum number of iterations.
%            tolx       - Error toleration for stopping condition.
%            Lips       - Lipschitz constant for gradient.
%            strcnvx    - Strong convexity parameter of f(x).
%*************************** LIONS@EPFL ***********************************
function [x, info] = AGDR(fx, gradf, parameter)
    
    fprintf('%s\n', repmat('*', 1, 68));

    % Set the clock.
    time1       = tic;
    timestart   = toc(time1);
    
    % Initialize x, y, t and find the initial function value.
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
        
        % Compute x_{k+1}.
        x_next = y - 1/parameter.Lips * gradf(y);
        % Save the old value of f(x_k) and evaluate the new falue of f(x_{k+1}).
        old_f = fx(x);
        new_f = fx(x_next);
        % Compare the old_f(x) and new_f(x) to decide to restart or not.
        if old_f < new_f
            %% restart
            y = x;
            t = 1;
            x_next = y - 1/parameter.Lips * gradf(y);
        end
        % Update the next iteration.
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
