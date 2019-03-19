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
        self.data=data
        # dict: { ||w|| : [w, b]}
        optimization_dict = {}

        # transforms - to check every version of vector possible
        transforms = [[1, 1],
                      [1,-1],
                      [-1,1],
                      [-1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        #flush the memory
        all_data=None

        # take big steps in parabolic graph
        # once you know the minimum with these big steps, take smaller steps
        # smaller steps are expensive

        # support vectors yi(xi.w+b)=1

        step_sizes = [self.max_feature_value*0.1,
                      self.max_feature_value*0.01,
                      #starts getting expensive
                      self.max_feature_value*0.001]

        #we care more about precision of w than b
        #extremly expensive step
        b_range_multiple = 5
        # don't need to take small steps like w
        b_multiple=5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            print('step :: ', step)
            # starting point of parabola - the top
            w = np.array([latest_optimum, latest_optimum])
            print('w :: ', w)
            # till we haven't found the minimum
            # we can do this because of it's a convex problem
            optimized = False
            while not optimized:
                # numpy.arange([start, ]stop, [step, ]dtype=None)
                # transforming w
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,self.max_feature_value*b_range_multiple,step*b_multiple):
                    for transform in transforms:
                        print('b , w :: ', b, w)
                        print('transorm :: ', transform)
                        w_t=w*transform
                        print('w_t :: ', w_t)
                        print('setting found_option value to true')
                        found_option=True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                print(yi,' *( ',w_t,'.',xi,')+ ',b,'=',(yi*(np.dot(w_t, xi)+b)))
                                if not yi*(np.dot(w_t, xi)+b)>=1:
                                    print('found_option to false')
                                    found_option=False

                        print('final found_option value : ', found_option)
                        if found_option:
                            print('seting opt_dict key .... ',np.linalg.norm(w_t))
                            optimization_dict[np.linalg.norm(w_t)] = [w_t, b]
                            print('optimization_dict[',[np.linalg.norm(w_t)],']','=[',w_t,',',b,']')

                print('w[0] :: ', w[0])
                if w[0]<0:
                    optimized=True
                    print('step optimized')
                else:
                    print('w not less than 0 :: ',w)
                    w = w - step
                    print('taking a step down :: ', w)

        norms = sorted(n for n in optimization_dict)
        print('norms :: ', norms)
        opt_choice = optimization_dict[norms[0]]
        print('opt_choice :: ', opt_choice)
        self.w = opt_choice[0]
        self.b = opt_choice[1]
        latest_optimum = opt_choice[0][0]+step*2


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


svm = support_vector_machine()
svm.fit(data=data_dict)