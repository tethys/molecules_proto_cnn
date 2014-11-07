setSeed(1)
%% Exercise 1
E_red = [0, 1/3, 0, 1;
      0, 0, 1/2, 0;
      1/2, 1/3, 0, 0;
      1/2, 1/3, 1/2, 0];
 
x0 = rand(4,1);
x0 = x0/sum(x0);


E_green = [0, 1/3, 0 ,    0;
          0,   0,   1/2,  0;
          1/2, 1/3, 0,   0;
          1/2, 1/3, 1/2, 0];

E_blue = zeros(6,6);
E_blue(1:4,1:4) =  E_red;
E_blue(5,6) = 1;
E_blue(6,5) = 1;

%% Exercise 2
M_red = create_pagerank_matrix(E_red);
v_red = power_iteration(M_red, x0);
[v,~] = eig(M_red);
fprintf('RED\n')
disp(v)
disp(eig(M_red))
disp(v_red)

fprintf('GREEN\n')
M_green = create_pagerank_matrix(E_green);
v_green = power_iteration(M_green, x0);
[v,~] = eig(M_green);
disp(v)
disp(eig(M_green))
disp(v_green)


fprintf('BLUE\n')
M_blue = create_pagerank_matrix(E_blue);
v_blue = power_iteration(M_blue, x1);
[v,d] = eig(M_blue);
disp(v)
disp(eig(M_blue))
disp(v_blue)