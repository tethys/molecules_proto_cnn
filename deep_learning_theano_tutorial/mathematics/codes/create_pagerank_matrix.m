function M = create_pagerank_matrix(E)
    p = 0.15;
    N = size(E, 1);
    M = (1-p)* E + p/N* ones(N,N);
    
end