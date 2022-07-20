from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import svm

from matplotlib import pyplot as plt

import vectorize

def train_models(tfidf, list_personality):
    type_indicators = ["IE: Introversion (I) - Extroversion (E)", "NS: Intuition (N) – Sensing (S)",
                       "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"]

    X = tfidf
    # kernals = ["poly", "rbf", "sigmoid", "linear"]
    xlabels = ["IE", "NS", "FT", "JP"]
    kernals = ["linear"]
    res = []
    for kernal in kernals:
        tmp = []
        for l in range(len(type_indicators)):
            print("%s ..." % (type_indicators[l]))

            Y = list_personality[:, l]

            seed = 3
            test_size = 0.33
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # define SVM model
            model = svm.SVC(decision_function_shape='ovr', kernel=kernal)
            # train the model
            model = model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(y_test, predictions)
            print("- %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))
        res.append(tmp)


def draw_res(kernals, res, n, xlabels):
    x = np.arange(n)
    width = 0.1

    plt.figure(figsize=(10, 6))
    for i, kernal in enumerate(kernals):
        plt.bar(x + i * width, res[i], width=width, label=kernal)
        plt.legend()

    plt.xticks(x + width*1.5, xlabels)
    plt.show()



def main():
    tfidf, list_personality = vectorize.tfidf()
    train_models(tfidf, list_personality)

if __name__ == "__main__":
    main()