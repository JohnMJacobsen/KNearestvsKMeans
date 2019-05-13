from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from pandas.plotting import parallel_coordinates
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import time
import HelperFunctions


class MyKNeighborsClassifier:

    def __init__(self, n_neighbors, df):
        self.n_neighbors = n_neighbors
        self.df = df

    def generate_sample(self):
        full_data = self.df.astype(float).values.tolist()
        random.shuffle(full_data)
        test_size = 0.2
        train_set = {2: [], 4: []}
        test_set = {2: [], 4: []}
        train_data = full_data[:-int(test_size * len(full_data))]
        test_data = full_data[-int(test_size * len(full_data)):]

        for i in train_data:
            train_set[i[-1]].append(i[:-1])

        for i in test_data:
            test_set[i[-1]].append(i[:-1])

        correct = 0
        total = 0

        for group in test_set:
            for data in test_set[group]:
                vote = k_nearest_neighbors(train_set, data, k=self.n_neighbors)
                if group == vote:
                    correct += 1
                total += 1
        accuracy = correct / total
        return accuracy


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

    return vote_result


def determine_optimal_k(x, y):
    k_bests = []
    for i in range(5):
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


def single_prediction(x, y, x_train, k):

    n_samples = x.shape[0]
    distances = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        distances[i] = (x_train - x[i]).dot(x_train - x[i])

    distances = np.c_[distances, y]
    sorted_distances = distances[distances[:, 0].argsort()]
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
            .format(k, round((sk_learn_end_time - sk_learn_start_time) / len(sklearn_iris_accuracies)), 6))
    average_iris_accuracy = round(sum(sklearn_iris_accuracies) / len(sklearn_iris_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_iris_accuracies), average_iris_accuracy))


def output_breast_cancer_data(f, x, y, k):
    f.write("BREAST CANCER DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_bc_accuracies = sklearn_k_nearest(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} neighbors: {} seconds\n"
            .format(k, round((sk_learn_end_time - sk_learn_start_time) / len(sklearn_bc_accuracies)), 6))
    average_bc_accuracy = round(sum(sklearn_bc_accuracies) / len(sklearn_bc_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_bc_accuracies), average_bc_accuracy))


def output_titanic_data(f, x, y, k):
    f.write("TITANIC DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_titanic_accuracies = sklearn_k_nearest(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} neighbors: {} seconds\n"
            .format(k, round((sk_learn_end_time - sk_learn_start_time) / len(sklearn_titanic_accuracies)), 6))
    average_titanic_accuracy = round(sum(sklearn_titanic_accuracies) / len(sklearn_titanic_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_titanic_accuracies), average_titanic_accuracy))


def output():

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

    my_iris_classifier = MyKNeighborsClassifier(ideal_iris_k, iris_df)
    # my_breast_cancer_classifier = MyKNeighborsClassifier(ideal_breast_cancer_k, breast_cancer_df)
    # my_titanic_classifier = MyKNeighborsClassifier(ideal_titanic_k, titanic_df)

    accuracy = my_iris_classifier.generate_sample()
    print(accuracy)


output()
