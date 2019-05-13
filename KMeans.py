from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from pandas.plotting import parallel_coordinates
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import HelperFunctions

class MyKMeansClassifier:
    def __init__(self, k, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                for centroid in self.centroids:
                    distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def custom_k_means(x, y, optimal_k):
    calculated_accuracies = []
    for i in range(50):
        correct = 0
        for i in range(len(x)):
            predict_me = np.array(x[i].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction = predict(predict_me)
            if prediction == y[i]:
                correct += 1

        accuracy = correct / len(x)
        if accuracy < 0.5:
            accuracy = 1.0 - accuracy
        calculated_accuracies.append(accuracy)


def sklearn_k_means(x, y, optimal_k):
    calculated_accuracies = []
    for i in range(50):
        correct = 0
        clf = KMeans(n_clusters=optimal_k)
        clf.fit(x)
        for i in range(len(x)):
            predict_me = np.array(x[i].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction = clf.predict(predict_me)
            if prediction[0] == y[i]:
                correct += 1
        accuracy = correct / len(x)
        if accuracy < 0.5:
            accuracy = 1.0 - accuracy
        calculated_accuracies.append(accuracy)
    return calculated_accuracies


def determine_optimal_k(x, y):
    k_bests = []
    for i in range(5):
        k_list = list(range(1, 8))
        scores = []

        for k in k_list:
            correct = 0
            clf = KMeans(n_clusters=k)
            clf.fit(x)
            for i in range(len(x)):
                predict_me = np.array(x[i].astype(float))
                predict_me = predict_me.reshape(-1, len(predict_me))
                prediction = clf.predict(predict_me)
                if prediction[0] == y[i]:
                    correct += 1
            accuracy = correct / len(x)
            if accuracy < 0.5:
                accuracy = 1.0 - accuracy
            scores.append(accuracy)

        mse = [1 - x for x in scores]
        # display_optimal_k_determination_visual(k_list, MSE)
        best_k = k_list[mse.index(min(mse))]
        k_bests.append(best_k)
    return max(set(k_bests), key=k_bests.count)


def prepare_output_file():
    f = open("KNearestResults.txt", "a")
    return f


def output_iris_data(f, x, y, k):
    f.write("IRIS DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_iris_accuracies = sklearn_k_means(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} means: {} seconds\n"
            .format(k, ((sk_learn_end_time - sk_learn_start_time) / len(sklearn_iris_accuracies))))
    average_iris_accuracy = round(sum(sklearn_iris_accuracies) / len(sklearn_iris_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_iris_accuracies), average_iris_accuracy))


def output_breast_cancer_data(f, x, y, k):
    f.write("BREAST CANCER DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_bc_accuracies = sklearn_k_means(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} means: {} seconds\n"
            .format(k, ((sk_learn_end_time - sk_learn_start_time) / len(sklearn_bc_accuracies))))
    average_bc_accuracy = round(sum(sklearn_bc_accuracies) / len(sklearn_bc_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_bc_accuracies), average_bc_accuracy))


def output_titanic_data(f, x, y, k):
    f.write("TITANIC DATA SET RESULTS\n")
    sk_learn_start_time = time.time()
    sklearn_titanic_accuracies = sklearn_k_means(x, y, k)
    sk_learn_end_time = time.time()
    f.write("Average sklearn time with {} means: {} seconds\n"
            .format(k, ((sk_learn_end_time - sk_learn_start_time) / len(sklearn_titanic_accuracies))))
    average_titanic_accuracy = round(sum(sklearn_titanic_accuracies) / len(sklearn_titanic_accuracies), 3)
    f.write("Average sklearn accuracy after {} runs: {}%\n"
            .format(len(sklearn_titanic_accuracies), average_titanic_accuracy))


def output():

    f = prepare_output_file()
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

    my_iris_classifier = MyKMeansClassifier(ideal_iris_k)
    my_breast_cancer_classifier = MyKMeansClassifier(ideal_breast_cancer_k)
    my_titanic_classifier = MyKMeansClassifier(ideal_titanic_k)

    my_iris_classifier.fit(iris_x)
    my_breast_cancer_classifier.fit(breast_cancer_x)
    my_titanic_classifier.fit(titanic_x)

    calculated_accuracies = []
    for i in range(50):
        correct = 0
        for i in range(len(iris_x)):
            predict_me = np.array(iris_x[i].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction = my_iris_classifier.predict(predict_me)
            if prediction == iris_y[i]:
                correct += 1

        accuracy = correct / len(iris_x)
        if accuracy < 0.5:
            accuracy = 1.0 - accuracy
        calculated_accuracies.append(accuracy)

    print(round(sum(calculated_accuracies) / len(calculated_accuracies), 3))

    # my_y_pred = my_iris_classifier.predict(iris_x)
    # print(my_y_pred)
    # accuracy = accuracy_score(y_test, my_y_pred)*100
    # print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')


output()
