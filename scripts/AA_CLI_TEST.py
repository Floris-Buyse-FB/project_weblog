import pandas as pd
import pickle
import os
import math
from sklearn.metrics import accuracy_score

def menu():
    link = str(input("Provide a link to a .csv file with the data you want to predict on:"))
    return link

def read_clean_data(link):
    df = pd.read_csv(link)
    df = df.drop(['NIGHT', 'ID', 'OTHER_METHOD'], axis=1)
    X = df.loc[:, df.columns != 'ROBOT']
    y = df['ROBOT']
    X = X.fillna(0)
    return X, y

def predict_robot(data, y):
    # load all .sav models from directory modellen
    for file in os.listdir("modellen"):
        if file.endswith(".sav"):
            model = pickle.load(open("./modellen/" + file, 'rb'))
            print(f"Model loaded {(file)}\n---------------------")
            print("Predicting...\n-------------")
            prediction = model.predict(data)
            perc_hum = (prediction == 0).sum() / len(prediction) * 100
            perc_rob = (prediction == 1).sum() / len(prediction) * 100
            print("{:.2f}% Human | {:.2f}% Robot".format(perc_hum, perc_rob))
            print("Accuracy: {:.2f}".format(((accuracy_score(y, prediction)) * 100)) + "%")
            print("\n----------------")


def main():

    link = menu()

    X, y = read_clean_data(link)

    predict_robot(X, y)

if __name__ == "__main__":

    main()