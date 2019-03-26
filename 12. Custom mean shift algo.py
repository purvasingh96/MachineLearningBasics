import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colors = 10*["g", "r", "c", "b", "k"]

# step 1 - assign every data point, a cluster center
# step-2 - take all data points, within that cluster center's bandwidth
# step-3 - take mean of all those data points
# step-4 - this is the new cluster center
# step-5 - optimum level is reached when cluster centers stop moving

class Custom_Mean_Shift:
    def __init__(self, bandwidth=4):
        self.bandwidth = bandwidth

    def fit(self, data):
        # assigning each data point, a centroid
        centroids = {}

        for i in range(len(data)):
            centroids[i]=data[i]

        while True:
            new_centroids = []
            for i in centroids:
                within_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.bandwidth:
                        within_bandwidth.append(featureset)

                # axis along which mean is computed
                # e.g let a =  np.array([[1, 2], [3, 4]])
                # np.mean(a) = 2.5 -> [(1+3)/2, (2+4)/2] = [2, 3] = (2+3)/2 = 2.5
                #np.mean(a, axis=0) = ([2., 3.])

                print("within_bandwidth :: ", within_bandwidth)
                new_centroid = np.average(within_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
                print("new_centroids :: ", new_centroids)

            # set removes duplicates
            uniques = sorted(list(set(new_centroids)))
            print("uniques :: ", uniques)

            prev_centroids = dict(centroids)
            print("prev centroids :: ", prev_centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            print("centroids array :: ", centroids)
            optimized=True

            for i  in centroids:
                if not  np.array_equal(centroids[i], prev_centroids[i]):
                    optimized=False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

def predict():
    pass

clf = Custom_Mean_Shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:, 0], X[:, 1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()
