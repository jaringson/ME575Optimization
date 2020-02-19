import math

class Dual:

    def __init__(self, value, deriv):
        self.value = value
        self.deriv = deriv

    def __str__(self):
        return "value: {0}\nderiv: {1}".format(self.value, self.deriv)

    def __add__(self, y):
        return Dual(self.value + y.value, self.deriv + y.deriv)

    def __pow__(self, p):
        return Dual(self.value**p, p*self.value**(p-1) * self.deriv)

    # def __truediv__(self, y):
    #     return Dual(self.value/y.value, self.deriv*(1.0/y.value)+self.value*(-y.value**-2)*y.deriv)

    def __truediv__(self, y):
        return Dual(self.value/y.value, (y.value*self.deriv - self.value*y.deriv) / y.value**2)

def sin(x):
    return Dual(math.sin(x.value), math.cos(x.value)*x.deriv)

def cos(x):
    return Dual(math.cos(x.value), -math.sin(x.value)*x.deriv)

def exp(x):
    return Dual(math.exp(x.value), math.exp(x.value)*x.deriv)

def sqrt(x):
    return Dual(math.sqrt(x.value), 0.5*x.value**(-0.5)*x.deriv)


def func(x):
    return exp(x)/sqrt(sin(x)**3 + cos(x)**3)


x = Dual(2.0, 1.0)
y = func(x)
print(y)
# exact derivative = 18.880984872278777
