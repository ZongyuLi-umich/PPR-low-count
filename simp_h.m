r = -50:0.0011:50;
s = -0.2;
b = 8;
y = 6;
c = s^2 + b;
f = simplify_h(r,s,c);
x = (b + sqrt(b^2 + b*s^2)) / abs(s);
my_curv = (x^2-b)/(x^2+b)^2;
plot(r,f);
hold on
line([-50,50],[1/(8*b),1/(8*b)],'Color','red','LineStyle','--')
hold on
line([-50,50],[my_curv,my_curv],'Color','blue','LineStyle','--')
hold off
(max(f,[],'all')*y+1)*2
2 + y / (4*b)
function f = simplify_h(t,s,c)
    f = (log(c) - log(t.^2+2*t*s+c))./(t.^2) + 2*s./(c.*t);
end
