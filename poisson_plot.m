t = -1000:0.0011:1000;
y = 6;
b = 2;
s = -20:0.009:20;
max_curv = (2 + y / (4*b)) * ones(length(s),1);
opt_curv = zeros(length(s),1);
my_curv = zeros(length(s),1);
for i = 1:length(s)
    c = s(i)^2 + b;
    opt_curv(i) = (max(simplify_h(t,s(i),c),[],'all')*y+1)*2;
    x = (b + sqrt(b^2+b*s(i)^2)) / s(i);
    my_curv(i) = curv_poisson(x,y,b);
end

figure;
plot(s,max_curv,'Color','black','LineStyle','--');
hold on
plot(s,my_curv,'Color','red');
hold on
plot(s,opt_curv,'Color','blue');
xlabel('s');
ylabel('curvature');
ylim([2 2.8]);
legend('max curvature','proposed curvature','optimal curvature (empirically)')
hold off
% opt_curv = (max(simplify_h(r,s,c),[],'all')*y+1)*2;
% x = (b + sqrt(b^2+b*s.^2)) ./ s;
% my_curv = curv_poisson(x,y,b);

%%
figure
diff = my_curv - opt_curv;
plot(s,diff)
xlabel('s');
ylabel('Proposed curvature minus optimal curvature')
ylim([0 0.12]);
% max_quad = 0.5 * max_curv .* (r-s).^2 + grad_poisson(s,y,b).*(r-s)+...
%             + poisson_func(s,y,b);
% opt_quad = 0.5 * opt_curv .* (r-s).^2 + grad_poisson(s,y,b).*(r-s)+...
%             + poisson_func(s,y,b);
% my_quad = 0.5 * my_curv .* (r-s).^2 + grad_poisson(s,y,b).*(r-s)+...
%             + poisson_func(s,y,b);
% plot(r,poisson_func(r,y,b),'Color','blue')
% hold on
% plot(r,max_quad, 'Color','green')
% hold on
% plot(r,opt_quad, 'Color','black')
% hold on
% plot(r,my_quad, 'Color','red')
% legend('orginial function','max curvature','optimal curvature','my curvature')
% sum((my_quad-poisson_func(r,y,b))<0)
function out = poisson_func(r,y,b)
    out = r.^2 + b - y * log(r.^2 + b);
end

function out = grad_poisson(r,y,b)
    out = 2 .* r * (1-y/(r.^2+b));
end

function out = curv_poisson(r,y,b)
    out = 2 + 2*y*(r.^2-b) ./((r.^2 + b).^2);
end

function f = simplify_h(t,s,c)
    f = (log(c) - log(t.^2+2*t*s+c))./(t.^2) + 2*s./(c.*t);
end