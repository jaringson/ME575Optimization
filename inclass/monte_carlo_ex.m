clear
clc

n = 10000;
fv = [];
gv = [];

for i=1:n
    x1 = 1;
    x2 = 1 + randn * 0.006;
    x3 = 1 + randn * 0.2;
    x = [x1, x2, x3];
   fv = [fv, f(x)];
   gv = [gv ,g(x)];
end

mean(fv)
std(fv)
histogram(fv)
figure
histogram(gv)

function val = f(x)
    val = x(1)^2+2*x(2)+3*x(3);
end

function val = g(x)
    val = x(1)+x(2)+x(3) - 3.5;
end