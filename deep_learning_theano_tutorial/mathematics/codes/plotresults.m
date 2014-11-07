%**************************************************************************
%*************************** LIONS@EPFL ***********************************
%**************************************************************************

set(0, 'DefaultAxesFontSize', 14);


% With respect to number of iterations.

hiter = figure(1); 
hold off
clear legends;

methodsdone = fieldnames(x);    
allMarkers  = {'+';'o';'*';'x';'square';'diamond';'v';'^';'>';'<';'pentagram';'hexagram'};
colors      = distinguishable_colors(numel(methodsdone));

if exist('fmin','var')
    ylbl = '$ \big( f(\mathbf{x}) - f^* \big) / f^* $';
else
    fmin = [];
    for myCount = 1:numel(methodsdone)
        fmin = [fmin; min(info.(methodsdone{myCount}).fx)];
    end
    fmin = min(fmin);
    ylbl = '$ \big( f(\mathbf{x}) - f^*_{aprx} \big) / f^*_{aprx} $';
    fprintf('f^* is approximated by the minimum over the methods you have implemented. \n');
end
if fmin == 0
    ylbl = '$f(\mathbf{x})$';
    for myCount = 1:numel(methodsdone)
        info.(methodsdone{myCount}).error = info.(methodsdone{myCount}).fx;
    end
    fprintf('f^* is 0 so you will get f(x) vs number of iterations and time. \n');
else
    for myCount = 1:numel(methodsdone)
        info.(methodsdone{myCount}).error = (info.(methodsdone{myCount}).fx - fmin)/fmin;
    end
end



for myCount = 1:numel(methodsdone)
    [xmarker, ymarker] = putmarker([], info.(methodsdone{myCount}).error, 5); 
    semilogy( xmarker, ymarker, allMarkers{myCount}, 'Color', colors(myCount,:), 'LineWidth', 3, 'MarkerSize', 16); hold on; 
end
for myCount = 1:numel(methodsdone)
semilogy ( info.(methodsdone{myCount}).error, 'Color', colors(myCount,:), 'LineWidth', 2); hold on;
end
xlabel('Number of iterations', 'Interpreter', 'latex', 'FontSize', 18);
ylabel(ylbl,'Interpreter', 'latex', 'FontSize', 18);
h1 = legend(methodsdone);
set(h1, 'Interpreter', 'latex', 'FontSize', 18);


% With respect to time.

htime = figure(2);  
hold off
for myCount = 1:numel(methodsdone)
    [xmarker, ymarker] = putmarker(info.(methodsdone{myCount}).time, info.(methodsdone{myCount}).error, 5); 
    semilogy( xmarker, ymarker, allMarkers{myCount}, 'Color', colors(myCount,:), 'LineWidth', 3, 'MarkerSize', 16); hold on; 
end
for myCount = 1:numel(methodsdone)
semilogy ( info.(methodsdone{myCount}).time, info.(methodsdone{myCount}).error, 'Color', colors(myCount,:), 'LineWidth', 2); hold on;
end
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel(ylbl,'Interpreter', 'latex', 'FontSize', 18);
h2 = legend(methodsdone);
set(h2, 'Interpreter', 'latex', 'FontSize', 18);

%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
