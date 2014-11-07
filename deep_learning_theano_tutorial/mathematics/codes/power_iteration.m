
function out_vector = power_iteration(E, x0)
Epow = E;
for i=1:15
    res = Epow*x0;
    res = res/norm(res);

    Epow = Epow * E;
end
out_vector = res/norm(res);
end