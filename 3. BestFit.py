# y = mx+b
# m = [mean(x).mean(y) - mean(x.y)]/[(mean(x))^2 - mean(x^2)]
#b = mean(y) - m(mean(x))

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# positively correlated data

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

# calculating m

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys)-m*mean(xs)

    return m, b

m, b = best_fit_slope_and_intercept(xs,ys)
print(m,b)

regression_line_y = [((m*x)+b) for x in xs]

plt.scatter(xs, ys)
plt.plot(xs, regression_line_y)
plt.show()