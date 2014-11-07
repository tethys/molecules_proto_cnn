setSeed(1)
%% Exercise 1
E_red = [0, 1/3, 0, 1;
      0, 0, 1/2, 0;
      1/2, 1/3, 0, 0;
      1/2, 1/3, 1/2, 0];
 
x0 = rand(4,1);
x0 = x0/sum(x0);

%% eigenvalues of E0
%% 
%   1.0000          
%  -0.5000 + 0.2887i
%  -0.5000 - 0.2887i
%  -0.0000  
% %  Columns 1 through 3
% 
%    0.6652            -0.2887 + 0.5000i  -0.2887 - 0.5000i
%    0.1996            -0.4330 - 0.2500i  -0.4330 + 0.2500i
%    0.3991             0.5774             0.5774          
%    0.5987             0.1443 - 0.2500i   0.1443 + 0.2500i
% 
%   Column 4
% 
%    0.5345          
%   -0.8018          
%   -0.0000          
%    0.2673  
% %
v_red = power_iteration(E_red, x0);
fprintf('Ex0 red converges to\n')
disp(v_red)

E_green = [0, 1/3, 0 ,    0;
          0,   0,   1/2,  0;
          1/2, 1/3, 0,   0;
          1/2, 1/3, 1/2, 0];
      
v_green = power_iteration(E_green, x0);
fprintf('Ex0 green converges to\n')
disp(v_green)
  

eig(E_green)
% v =
% 
%         0            -0.2226            -0.3757 - 0.3534i  -0.3757 + 0.3534i
%         0            -0.3748             0.5962             0.5962          
%         0            -0.4208            -0.3347 + 0.3147i  -0.3347 - 0.3147i
%    1.0000            -0.7956             0.2615 + 0.3147i   0.2615 - 0.3147i
% 
% 
% d =
% 
%         0                  0                  0                  0          
%         0             0.5614                  0                  0          
%         0                  0            -0.2807 + 0.2640i        0          
%         0                  0                  0            -0.2807 - 0.2640i  
  
E_blue = zeros(6,6);
E_blue(1:4,1:4) =  E_red;
E_blue(5,6) = 1;
E_blue(6,5) = 1;
x1 = rand(6,1);
x1 = x1/ sum(x1); 
[v,d] = eig(E_blue);
v_blue = power_iteration(E_blue, x1);
fprintf('Blur matrix results\n')
disp(v_blue)
disp(eig(E_blue))
disp(v)












  