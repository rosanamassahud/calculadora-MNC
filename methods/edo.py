def f1(x, y):
    return (1 - y / x)


def runge1(f, x0, y0, xf, h):
    y1 = y0 + h * f(x0, y0)
    x1 = x0 + h
    n = int((xf - x0) / h)
    for i in range(1, n):
        x0 = x1
        y0 = y1
        y1 = y0 + h * f(x0, y0)
        x1 = x0 + h
    return x1, y1


def runge2(f, x0, y0, xf, h):
    y1 = y0 + h / 2 * (f(x0, y0) + f(x0 + h, y0 + h * f(x0, y0)))
    x1 = x0 + h
    n = int((xf - x0) / h)
    for i in range(1, n):
        x0 = x1
        y0 = y1
        y1 = y0 + h / 2 * (f(x0, y0) + f(x0 + h, y0 + h * f(x0, y0)))
        x1 = x0 + h

    return x1, y1


if(__name__=='__main__'):
    f = f1
    x0 = 2
    y0 = 2
    xf = 2.1
    h = 0.05
    x1, y1 = runge1(f, x0, y0, xf, h)
    print("Runge 1: y[", x1, "] = ", y1)
    x1, y1 = runge2(f, x0, y0, xf, h)
    print("Runge 2: y[", x1, "] = ", y1)