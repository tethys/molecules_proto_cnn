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
    Larray = zeros(parameter.maxit,1);
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
        
         % Evaluate the function and the gradients.
        % Approximate local Lipschitz constant by line-search.
        L = parameter.Lips/8;
        d = gradf(y);
        nrm_d2 = d'*d;
        kappa = 1;
        for j=1:50
            if fx(y - 1/L* gradf(y)) <= fx(y) - kappa*0.5/L*nrm_d2;
                break;
            end
            L = L*2;
        end
        % Update the next iteration. local Lipschitz constant.
        alpha = 1/L;
        % Update the next iteration.
        
        x_next = y - alpha * gradf(y);
        Larray(iter) = L;
  
        old_f = fx(x);
        new_f = fx(x_next);
        % Compare the old_f(x) and new_f(x) to decide to restart or not.
        if old_f < new_f
            %% restart
            y = x;
            t = 1;
            x_next = y - alpha * gradf(y);
        end
        
        % Restart the iteration if necessary.

        if (iter >= 2)
            theta_next = Larray(iter)/ Larray(iter - 1);
        else
            theta_next =  (Larray(iter)*8)/parameter.Lips;
        end
        t_next = 0.5* (1 + sqrt(1+ 4* theta_next*t*t));
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
