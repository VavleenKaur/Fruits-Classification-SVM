# Vavleen Kaur
# Fruit classification using SVM
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC


def importDataset():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/VavleenKaur/Fruits-Classification-SVM/main/fruits.csv")
    df = pd.DataFrame(data=data)
    print(df.info())
    label_encoder = preprocessing.LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])
    df['Class'].unique()
    x = df.iloc[:, 1:3]
    y = df.iloc[:, 3]
    return x, y, df


def heatmap(df):
    sns.heatmap(df.iloc[:, :5].corr(), annot=True)
    plt.show()


def MissingNo(df):
    msn.bar(df)
    plt.show()


def trainTestSplit(x, y):
    return train_test_split(x, y, random_state=42, test_size=0.3)


def trainModel(x_train, y_train):
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model


def testModel(model, x_test):
    return model.predict(x_test)


def cnMatrix(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    print(f'Accuracy is : %', accuracy*100)
    print(f'Precision is : ', precision)
    print(f'Recall is : ', recall)


def main():
    x, y, df = importDataset()
    heatmap(df)
    MissingNo(df)
    x_train, x_test, y_train, y_test = trainTestSplit(x, y)
    model = trainModel(x_train, y_train)
    y_pred = testModel(model, x_test)
    cnMatrix(y_test, y_pred)
    print("Confusion Matrix-")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report-")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
