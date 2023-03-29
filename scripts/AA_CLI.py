import pandas as pd
import pickle
import os

def menu():
    link = str(input("Provide a link to a .csv file with the data you want to predict on:"))
    return link

def predict_robot(data):
    # load all .sav models from directory modellen
    for file in os.listdir("modellen"):
        if file.endswith(".sav"):
            model = pickle.load(open("../modellen/" + file, 'rb'))
            print("Model loaded")
            print("Predicting...")
            print(model.predict(data))

def main():
    link = menu()
    data = pd.read_csv(link)
    predict_robot(data)

if __name__ == "__main__":

    main()