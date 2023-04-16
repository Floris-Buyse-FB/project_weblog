import pandas as pd
import pickle
import os
import math
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def menu():
    link = str(input("Provide a link to a .csv file with the data you want to predict on:"))
    return link

def read_clean_data(link):
    df = pd.read_csv(link)
    df = df.drop(['HTTP_RESPONSE_2XX', 'GET_METHOD', 'ID', 'STANDARD_DEVIATION', 'SF_REFERRER', 'SF_FILETYPE', 'OTHER_METHOD', 'POST_METHOD', 'HEAD_METHOD', 'HTTP_RESPONSE_3XX',  'HTTP_RESPONSE_4XX', 'HTTP_RESPONSE_5XX','REPEATED_REQUESTS'], axis=1)
    X = df.drop(['ROBOT'], axis=1)
    y = df['ROBOT']
    return X, y

def predict_robot(data, y):
    # load all .sav models from directory modellen
    for file in os.listdir("main_models"):
        if file.endswith("1.sav"):
            model = pickle.load(open("main_models/" + file, 'rb'))
            print(f"Model loaded {(file)}\n---------------------")
            print("Predicting...\n-------------")
            if file == "kneighbors_classifier1.sav" or file == "logistic_regression1.sav" or file == "linear_svc1.sav":
                data1 = scaler.fit_transform(data)
                prediction = model.predict(data1)
            else:
                prediction = model.predict(data)
            perc_hum = (prediction == 0).sum() / len(prediction) * 100
            perc_rob = (prediction == 1).sum() / len(prediction) * 100
            cm = confusion_matrix(y, prediction)
            print("{:.2f}% Human | {:.2f}% Robot".format(perc_hum, perc_rob))
            print("Accuracy: {:.2f}".format(((accuracy_score(y, prediction)) * 100)) + "%")
            print(f"Confusion matrix:\n{cm}")
            print("\n----------------")


def main():

    link = menu()

    X, y = read_clean_data(link)

    predict_robot(X, y)

if __name__ == "__main__":

    main()