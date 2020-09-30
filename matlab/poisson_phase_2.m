r = 0:0.01:20;
y = 1;
b = 2;
h = poisson_phase_retrieval(r,b,y);


% Might be relative to s^2 - 3b
plot(r,h)
function h = poisson_phase_retrieval(r, b, y)
    h = (exp(-r) + b) - y*log(exp(-r) + b);
end

