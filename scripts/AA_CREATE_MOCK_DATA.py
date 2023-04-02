import csv
import random
import pandas as pd
import numpy as np

df = pd.read_csv("data/weblogs.csv")

header = df.columns.tolist()

rows = df.values.tolist()

# generate random data for the columns based on the standard deviation and mean of the column except the last column should be 1 or 0
for row in rows:
    for i in range(len(row) - 1):
        if i == 0:
            continue
        row[i] = random.gauss(row[i], 1)

    row[-1] = random.randint(0, 1)

# write the data to a csv file
with open("data/test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

