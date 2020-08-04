from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from decisionAndRandomforecast import *

import vectorize

def train_models(tfidf, list_personality):
    type_indicators = ["IE: Introversion (I) - Extroversion (E)", "NS: Intuition (N) – Sensing (S)",
                       "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"]

    X = tfidf
    for l in range(len(type_indicators)):
        print("%s ..." % (type_indicators[l]))

        Y = list_personality[:, l]

        seed = 9
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # 训练模型
        # model = XGBClassifier()
        model = RandomForestClassifier(max_depth = 5)
        # model = DecisionTreeClassifier(max_depth = 3)
        model.fit(X_train, y_train)


        # 在测试集上进行预测
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("- %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))

def main():
    tfidf, list_personality = vectorize.tfidf()
    train_models(tfidf, list_personality)


if __name__ == "__main__":
    main()