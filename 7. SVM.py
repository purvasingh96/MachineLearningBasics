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
        self.data = data
        # dict: { ||w|| : [w, b]}
        optimization_dict = {}

        # transforms - to check every version of vector possible
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        # flush the memory
        all_data = None

        # take big steps in parabolic graph
        # once you know the minimum with these big steps, take smaller steps
        # smaller steps are expensive

        # support vectors yi(xi.w+b)=1
        # optimum value is the one whose value stays close to 1
        # for both +ve and -ve classes.

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # starts getting expensive
                      self.max_feature_value * 0.001]

        # we care more about precision of w than b
        # extremly expensive step
        b_range_multiple = 2
        # don't need to take small steps like w
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            # starting point of parabola - the top
            w = np.array([latest_optimum, latest_optimum])
            # till we haven't found the minimum
            # we can do this because of it's a convex problem
            optimized = False
            while not optimized:
                # numpy.arange([start, ]stop, [step, ]dtype=None)
                # transforming w
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transform in transforms:
                        w_t = w * transform
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                # constraint function :: yi( xi.wi + b )>=1
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            # np.linalg.norm :: finds magnitude of vector
                            optimization_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('optimized a step')
                else:
                    w = w - step


        norms = sorted([n for n in optimization_dict])
        opt_choice = optimization_dict[norms[0]]
        self.w = opt_choice[0]
        self.b = opt_choice[1]
        latest_optimum = opt_choice[0][0] + step * 2

    # classification based on sign
    # sign( x.w+b )
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        # scatter plot for data_dict
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # eq. of hyperplane :: y = x.w+b
        # positive support vector :: y=1
        # negative support vector :: y=-1
        # decision boundry :: y=0
        def hyperplane(x, w, b, y):
            return (-w[0] * x - b + y) / w[1]

        # used to limit our graph
        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b)=1
        # positive support vector hyperplane
        # psv1, psv2 :: y co-ordinates of hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # (w.x+b)=-1
        # negative support vector hyperplane
        # psv1, psv2 :: y co-ordinates of hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # (w.x+b)=0
        # decision boundry vector hyperplane
        # psv1, psv2 :: y co-ordinates of hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])

        plt.show()


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])
             }

svm = support_vector_machine()
svm.fit(data=data_dict)
svm.visualize()
