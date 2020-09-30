t = -10:0.01:10;
s = 8;
h  = log(1+exp(-t));
gh = grad_h(t);
curv = (grad_h(s)-grad_h(-s)) / (2*s);

plot(t,gh)
hold on
plot(t,curv .* t - 0.5);

function h = grad_h(t)
    h = -1 ./ (1+exp(t));
end