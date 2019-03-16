import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')


def custom_k_nearest_neighbors(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('k is set to value less than  total   groups')

    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    #[[2.0, 'r'], [2.23606797749979, 'r'], [3.1622776601683795, 'r']]
    print(sorted(distances)[:k])
    votes = [i[1] for i in sorted(distances)[:k]]
    #Counter library finds the majority vote
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)

plt.scatter(new_features[0], new_features[1], s=100)

result = custom_k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)
plt.show()