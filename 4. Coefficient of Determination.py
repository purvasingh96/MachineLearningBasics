# y = mx+b
# m = [mean(x).mean(y) - mean(x.y)]/[(mean(x))^2 - mean(x^2)]
#b = mean(y) - m(mean(x))

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

# positively correlated data

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

# calculating m

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys)-m*mean(xs)

    return m, b

m, b = best_fit_slope_and_intercept(xs,ys)

def squared_error(ys_original , ys_line):
    return sum((ys_line-ys_original)**2)

def coeff_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    print(squared_error_regr)
    print(squared_error_y_mean)

    r_squared = 1 - (squared_error_regr/squared_error_y_mean)

    return r_squared


def create_dataset(limit, variance, step=2, correlation=False):
    val=1
    ys=[]
    for i in range(limit):
        y =val+random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='neg':
            val-=step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)

xs, ys = create_dataset(40,10,2,correlation='pos')
regression_line_y = [((m*x)+b) for x in xs]

r_squared = coeff_of_determination(ys, regression_line_y)
print(r_squared)

plt.scatter(xs,ys,color='#003F72', label = 'data')
plt.plot(xs, regression_line_y, label = 'regression line')
plt.legend(loc=4)
plt.show()