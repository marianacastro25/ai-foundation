import csv

with open("winequality-red.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        print(i, line)
       
        
       