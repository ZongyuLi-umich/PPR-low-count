y = 6;
b = 2;
s = 8;
r = -15:0.001:15;

k1 = curv_poisson(s,y,b);
k2 = curv_poisson((b+sqrt(b^2+b*s^2))/s,y,b);
k3 = curv_poisson((b-sqrt(b^2+b*s^2))/s,y,b);
y1 = k1 * (r - s) + grad_poisson(s,y,b);
y2 = k2 * (r - s) + grad_poisson(s,y,b);
y3 = k3 * (r - s) + grad_poisson(s,y,b);
x1 = (b+sqrt(b^2+b*s^2))/s;
x2 = (b-sqrt(b^2+b*s^2))/s;
figure
plot(r,grad_poisson(r,y,b),'LineWidth',2)
hold on
plot(r,y1,'LineStyle','--','LineWidth',1)
hold on
plot(r,y2,'LineStyle','--','LineWidth',1)
hold on
plot(r,y3,'LineStyle','--','LineWidth',1)
hold on
plot(x1,grad_poisson(x1,y,b),'.-','MarkerSize',15,'MarkerEdgeColor','k',...
    'MarkerFaceColor',[1 1 1]);
hold on
plot(x2,grad_poisson(x2,y,b),'.-','MarkerSize',15,'MarkerEdgeColor','k','MarkerFaceColor',[1 1 1]);
hold on
plot(s,grad_poisson(s,y,b),'.-','MarkerSize',15,'MarkerEdgeColor','k','MarkerFaceColor',[1 1 1]);
% txt = '\fontsize{15} \leftarrow x_1';
% text(x1,grad_poisson(x1,y,b),txt);
% txt = '\fontsize{15} x_2 \rightarrow';
% text(x2,grad_poisson(x2,y,b),txt);
% txt = '\fontsize{15} \leftarrow s';
% text(s,grad_poisson(s,y,b),txt);
x = [0.74 0.69];    % adjust length and location of arrow 
y = [0.3 0.3];      % adjust hieght and width of arrow
annotation('textarrow',x,y,'String',' x_1 ','FontSize',13,'Linewidth',2)

x = [0.74 0.79];    % adjust length and location of arrow 
y = [0.3 0.3];      % adjust hieght and width of arrow
annotation('textarrow',x,y,'String',' x_2 ','FontSize',13,'Linewidth',2)

x = [0.74 0.69];    % adjust length and location of arrow 
y = [0.3 0.3];      % adjust hieght and width of arrow
annotation('textarrow',x,y,'String',' s ','FontSize',13,'Linewidth',2)
xlabel('r');
ylabel('grad h(r)');
hold off
function out = grad_poisson(r,y,b)
    out = 2 * r .* (1-y ./(r.^2+b));
end

function out = curv_poisson(r,y,b)
    out = 2 + 2*y*(r.^2-b) ./((r.^2 + b).^2);
end