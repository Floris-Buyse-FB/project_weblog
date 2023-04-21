import pickle
import pandas as pd
import os
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL = "main_models/random_forest_classifier2.sav"

def load_model():
    model = pickle.load(open(MODEL, 'rb'))
    return model

def menu():
    link = str(input("Provide a link to a .csv file with the data you want to predict on:"))
    return link

def read_clean_data(link):
    df = pd.read_csv(link)
    df = df.drop(['ID', 'STANDARD_DEVIATION', 'SF_REFERRER', 'SF_FILETYPE', 'OTHER_METHOD', 'POST_METHOD', 'HEAD_METHOD', 'HTTP_RESPONSE_3XX',  'HTTP_RESPONSE_4XX', 'HTTP_RESPONSE_5XX','REPEATED_REQUESTS'], axis=1)
    X = df.drop(['ROBOT'], axis=1)
    y = df['ROBOT']
    return X, y

def predict_robot(data, y):
    model = load_model()
    print(f"Model loaded {(MODEL)}\n---------------------")
    print("Predicting...\n-------------")
    prediction = model.predict(data)
    print(f"\n{prediction}\n")
    perc_hum = (prediction == 0).sum() / len(prediction) * 100
    perc_rob = (prediction == 1).sum() / len(prediction) * 100
    print("{:.2f}% Human | {:.2f}% Robot".format(perc_hum, perc_rob))
    print("Accuracy: {:.2f}".format(((accuracy_score(y, prediction)) * 100)) + "%")
    print("Confusion matrix:\n", np.round(confusion_matrix(y, prediction, normalize='true'), 2))

def retrain_model(X, y):
    retrain = str(input("Do you want to retrain the model? (y/n):"))
    while retrain != "y" and retrain != "n":
        retrain = str(input("Do you want to retrain the model? (y/n):"))
    
    if retrain == "y":
        print("Retraining model...")

        model = load_model()

        print(f"Model loaded {(MODEL)}\n---------------------")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        print("Accuracy: {:.2f}".format(((accuracy_score(y_test, pred)) * 100)) + "%")

        print("Confusion matrix:\n", confusion_matrix(y_test, pred))

        save_model = str(input("Do you want to save the model? (y/n):"))
        while save_model != 'y' and save_model != 'n':
            save_model = str(input("Do you want to save the model? (y/n):"))

        if save_model == "y":
            name = str(input("Provide a name for the new model:"))
            filename = "retrained_models/" + name 
            pickle.dump(model, open(filename, 'wb'))
            print("Model saved.")
        else:
            print("Model not saved.")
            print("Exiting...")

    else:
        print("Model not retrained.")
        print("Exiting...")

def main():
    
    link = menu()
    
    X, y = read_clean_data(link)
    
    predict_robot(X, y)

    retrain_model(X, y)

if __name__ == "__main__":
        
    main()