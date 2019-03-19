from scipy.optimize import minimize

y1 = 12
y2 = 37
y3 = 14
args = (y1, y2, y3)

x0 = 0


def objective(x, *args):
    y1, y2, y3 = args
    return (y1 * x + y2 * x * x + y3 * x * x * x)


optimal = minimize(objective, x0, args, method='CG')
print(optimal.x)
