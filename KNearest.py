from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pandas as pd
import numpy as np
import random
import time

fClear = open("KNearestResults.txt", "w")
fClear.write("")

f = open("KNearestResults.txt", "a")

sklearnIrisAccuracies = []
customAccuracies = []


def k_nearest_neighbors(data, predict, k=3):
    if len(data) > 3:
        print("Insufficient data points")
        exit(-1)

    distances = []
    for group in data:
        for features in data[group]:
            print(group)
            print(features)
            # euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))


def run_sklearn_k_nearest(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    sklearnIrisAccuracies.append(accuracy)


iris_df = pd.read_csv('iris.data.txt')
iris_y = np.array(iris_df['class'])
iris_df = iris_df.drop(['class'], 1)
iris_X = np.array(iris_df)

sklearnStartTime = time.time()
for i in range(25):
    run_sklearn_k_nearest(iris_X, iris_y)
skLearnEndTime = time.time()

f.write("Total sklearn time: {}\n".format(skLearnEndTime - sklearnStartTime))
f.write("Average sklearn time: {}\n".format((skLearnEndTime - sklearnStartTime) / len(sklearnIrisAccuracies)))

for i in range(len(sklearnIrisAccuracies)):
    f.write("{}: {}\n".format(i, sklearnIrisAccuracies[i]))

averageIrisAccuracy = sum(sklearnIrisAccuracies) / len(sklearnIrisAccuracies)
f.write("Average sklearn accuracy: {}".format(averageIrisAccuracy))
