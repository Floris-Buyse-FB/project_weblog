import pandas as pd
import pickle
import os
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

scaler = StandardScaler()

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
    for file in os.listdir("main_models"):

        model = pickle.load(open("main_models/" + file, 'rb'))
        print(f"Model loaded {(file)}\n---------------------")

        print("Predicting...\n-------------")

        # if file has scaled in its name, scale the data

        if file.endswith("scaled.sav"):
            data1 = scaler.fit_transform(data)
            prediction = model.predict(data1)
        else:
            prediction = model.predict(data)

        # try:
        #     df = pd.DataFrame({'Actual': y, 'Predicted': prediction})
        #     df['Probability Robot'] = model.predict_proba(data)[:,1]
        #     df['Probability Robot'] = df['Probability Robot'].apply(lambda x: round(x * 100, 2))
        #     df['Probability Robot'] = df['Probability Robot'].apply(lambda x: str(x) + "%")
        #     df['Probability Human'] = model.predict_proba(data)[:,0]
        #     df['Probability Human'] = df['Probability Human'].apply(lambda x: round(x * 100, 2))
        #     df['Probability Human'] = df['Probability Human'].apply(lambda x: str(x) + "%")
        #     df['Correct'] = df['Actual'] == df['Predicted']
        #     df['Correct'] = df['Correct'].apply(lambda x: "Yes" if x == True else "No")
        #     print(df)
        # except:
        #     pass

        perc_hum = (prediction == 0).sum() / len(prediction) * 100
        perc_rob = (prediction == 1).sum() / len(prediction) * 100

        cm = np.round(confusion_matrix(y, prediction, normalize='true'), 2)
        
        print("{:.2f}% Human | {:.2f}% Robot".format(perc_hum, perc_rob))
        print("Accuracy: {:.2f}".format(((accuracy_score(y, prediction)) * 100)) + "%")
        print("Classification report:\n", classification_report(y, prediction), "\n")
        print(f"Confusion matrix:\n{cm}")
        print("\n----------------")

def main():

    link = menu()

    X, y = read_clean_data(link)

    predict_robot(X, y)

if __name__ == "__main__":

    main()