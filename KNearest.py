from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from pandas.plotting import parallel_coordinates
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import HelperFunctions


class MyKNeighborsClassifier:

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        n_samples = X.shape[0]
        if self.n_neighbors > n_samples:
            raise ValueError("Insufficient data points")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")

        # finding and saving all possible class labels
        self.classes_ = np.unique(y)

        self.X = X
        self.y = y

    def predict(self, X_test):

        n_predictions, n_features = X_test.shape

        # allocationg space for array of predictions
        predictions = np.empty(n_predictions, dtype=int)

        # loop over all observations
        for i in range(n_predictions):
            # calculation of single prediction
            predictions[i] = single_prediction(self.X, self.y, X_test[i, :], self.n_neighbors)

        return predictions


def k_nearest_neighbors(data, predict, k):
    if len(data) > k:
        print("Insufficient data points")
        exit(-1)

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence


def determine_optimal_k(x, y):
    k_bests = []
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        k_list = list(range(1, 16, 2))

        cv_scores = []

        for k in k_list:
            knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())

        mse = [1 - x for x in cv_scores]
        # display_optimal_k_determination_visual(k_list, MSE)

        best_k = k_list[mse.index(min(mse))]
        k_bests.append(best_k)
    return max(set(k_bests), key=k_bests.count)


def sklearn_k_nearest(x, y, optimal_k):
    calculated_accuracies = []
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        clf = neighbors.KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=-1)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)
        calculated_accuracies.append(accuracy)
    return calculated_accuracies


def parallel_coordinates_data_visualization(dataset_df):
    plt.figure(figsize=(10, 8))
    parallel_coordinates(dataset_df, "class")
    plt.title('Parallel Coordinates Plot', fontsize=15, fontweight='bold')
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Features values', fontsize=15)
    plt.legend(loc=1, prop={'size': 15}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
    plt.show()


def display_optimal_k_determination_visual(k_list, mse):
    plt.figure()
    plt.figure(figsize=(15, 10))
    plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
    plt.xlabel('Number of Neighbors K', fontsize=15)
    plt.ylabel('Misclassification Error', fontsize=15)
    sns.set_style("whitegrid")
    plt.plot(k_list, mse)
    plt.show()


def single_prediction(X, y, x_train, k):

    # number of samples inside training set
    n_samples = X.shape[0]

    # create array for distances and targets
    distances = np.empty(n_samples, dtype=np.float64)

    # distance calculation
    for i in range(n_samples):
        distances[i] = (x_train - X[i]).dot(x_train - X[i])

    # combining arrays as columns
    distances = np.c_[distances, y]
    # sorting array by value of first column
    sorted_distances = distances[distances[:, 0].argsort()]
    # celecting labels associeted with k smallest distances
    targets = sorted_distances[0:k, 1]

    unique, counts = np.unique(targets, return_counts=True)
    return unique[np.argmax(counts)]


def restore_output_file():
    fclear = open("KNearestResults.txt", "w")
    fclear.write("")
    f = open("KNearestResults.txt", "a")
    return f


def output_iris_data(f, x, y, k):
    f.write("IRIS DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_iris_accuracies = sklearn_k_nearest(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} neighbors: {} seconds\n"
            .format(k, round((sk_learn_end_time - sk_learn_start_time) / len(sklearn_iris_accuracies)), 3))
    average_iris_accuracy = round(sum(sklearn_iris_accuracies) / len(sklearn_iris_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_iris_accuracies), average_iris_accuracy))


def output_breast_cancer_data(f, x, y, k):
    f.write("BREAST CANCER DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_bc_accuracies = sklearn_k_nearest(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} neighbors: {} seconds\n"
            .format(k, round((sk_learn_end_time - sk_learn_start_time) / len(sklearn_bc_accuracies)), 3))
    average_bc_accuracy = round(sum(sklearn_bc_accuracies) / len(sklearn_bc_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_bc_accuracies), average_bc_accuracy))


def output_titanic_data(f, x, y, k):
    f.write("TITANIC DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_titanic_accuracies = sklearn_k_nearest(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} neighbors: {} seconds\n"
            .format(k, round((sk_learn_end_time - sk_learn_start_time) / len(sklearn_titanic_accuracies)), 3))
    average_titanic_accuracy = round(sum(sklearn_titanic_accuracies) / len(sklearn_titanic_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_titanic_accuracies), average_titanic_accuracy))


def main():

    f = restore_output_file()
    iris_df, iris_x, iris_y = HelperFunctions.prepare_iris_data()
    breast_cancer_df, breast_cancer_x, breast_cancer_y = HelperFunctions.prepare_breast_cancer_data()
    titanic_df, titanic_x, titanic_y = HelperFunctions.prepare_titanic_data()

    # parallel_coordinates_data_visualization(iris_df)

    ideal_iris_k = determine_optimal_k(iris_x, iris_y)
    ideal_breast_cancer_k = determine_optimal_k(breast_cancer_x, breast_cancer_y)
    ideal_titanic_k = determine_optimal_k(titanic_x, titanic_y)

    output_iris_data(f, iris_x, iris_y, ideal_iris_k)
    output_breast_cancer_data(f, breast_cancer_x, breast_cancer_y, ideal_breast_cancer_k)
    output_titanic_data(f, titanic_x, titanic_y, ideal_titanic_k)

    # my_classifier = MyKNeighborsClassifier(ideal_k)
    # my_classifier.fit(iris_x, iris_y)

    # my_y_pred = my_classifier.predict(iris_x)
    # accuracy = accuracy_score(y_test, my_y_pred)*100
    # print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')


main()

"""
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print("Handmade Accuracy: {}".format(correct/total))
customAccuracies.append(correct/total)
"""
