import matplotlib.pyplot as plt
from matplotlib import style
import numpy    as np

style.use('ggplot')


class support_vector_machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # training our data
    def fit(self, data):
        pass

    # classification based on sign
    # sign( x.w+b )
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        return classification


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])

             }