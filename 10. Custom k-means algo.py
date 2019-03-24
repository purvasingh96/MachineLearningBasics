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
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)


colors =10*["g","r","c","b","k"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter

    def fit(self, data):
        self.centroids={}

        # initially setting centroids as
        # first two elements of array
        # i = 0, 1
        # centroids[0] = [1, 2]
        # centroids[1] = [1.5, 1.8]
        for i in range(self.k):
            self.centroids[i]=data[i]

        for i in range(self.max_iter):
            self.classifications={}

            # classifications[0] = []
            # classifications[1] = []
            for i in range(self.k):
                self.classifications[i]=[]

            for featureset in data:
                # 1st iteration
                # featureset = [1, 2]
                # self.centroids[0]=[1,2]
                # self.centroids[1]=[1.5, 1.8]

                # euclidean_distances = [ distance( [1, 2], [1, 2] ), distance([1, 2], [1.5, 1.8])
                euclidean_distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]

                # min. of euclidean_distances = distance( [1, 2], [1, 2] ) = 0
                # classifying data point [1, 2] in category index(min(euclidean_distances)), i.e 0
                # classification = 0
                classification = euclidean_distances.index(min(euclidean_distances))

                # appending [1, 2] in category 0
                self.classifications[classification].append(featureset)
                print("eucld. dist :: ", euclidean_distances)
                print('classification :: ', classification)
                print('appending featureset :: ', featureset, ':: ',self.classifications)

            prev_centroids = dict(self.centroids)
            print("prev_centroids :: ", prev_centroids)


            # after few more iterations
            # classifications set will have list of points in each category (0, 1)
            # classified on the basis of their eucledian distance from centroids
            # sample classifications set would look like this ::
            #  {0: [array([1., 2.]), array([1.5, 1.8]), array([1. , 0.6])],
            #  1: [array([5., 8.]), array([8., 8.]), array([ 9., 11.])]}

            # inorder to modify the centroids
            # for the next iteration
            # current_centroid would become mean of all data points in each category
            for classification in self.classifications:
                self.centroids[classification]=np.average(self.classifications[classification], axis=0)
                print("self.centroids after average :: ", self.centroids[classification])

            optimized = True

            # if current centroid's movement as compared to original centroid
            # is less than tolerance value
            # then we have found our final optimized centroid value

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

                if optimized:
                     break


def predict(self,data):
    distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
    classification = distances.index(min(distances))
    return classification


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()
