from staci import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statistics import mean
from sklearn.neural_network import MLPRegressor

datasets = ["bike", "servo", "housing", "auto", "concrete"]

depth = 4

for dataset in datasets:
    df = pd.read_csv("../datasets/" + dataset + ".csv", header=0)
    y = df["target"]
    X_raw = df.drop("target", axis=1)
    nominal_features = []
    numerical_features = []