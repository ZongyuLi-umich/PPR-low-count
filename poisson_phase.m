r = -50:0.001:50;
y = 4;
b = 6;
h = poisson_phase_retrieval(r,b,y);
s = 1;
x = (b+sqrt(b^2+b*s^2))/s;
curv = curv_poisson(x,b,y);


grad_s = grad_poisson(s,b,y);
H = poisson_phase_retrieval(s,b,y) + grad_s .*(r-s) + ...
    (1+y/(8*b))*(r-s).^2;
% Might be relative to s^2 - 3b
% plot(r,h)
% hold on
% plot(r,H)
% legend('h','H')
% hold off
sum((H-h)<0);
grad_r = grad_poisson(r,b,y);
figure
plot(r,grad_r);
pt1 = grad_poisson(s,b,y);

% k = curv_poisson(s,b,y);
f = curv .* (r - s) + pt1;
hold on
plot(r,f);
hold off
2 + y / (4*b)
% curv_r = curv_poisson(r,b,y);
% figure
% plot(r,curv_r);
% (grad_poisson(s,b,y)-grad_poisson(-s,b,y))/(2*s)
function h = poisson_phase_retrieval(r, b, y)
    h = (r.^2 + b) - y*log(r.^2 + b);
end
function h = grad_poisson(r,b,y)
    h = 2 * r.*(1-y./(r.^2+b));
end
function h = curv_poisson(r,b,y)
    h = 2 + 2*y*(r.^2-b)./((r.^2+b).^2);
end
