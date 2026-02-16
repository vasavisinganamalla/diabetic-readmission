#Loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

#loading Dataset
df = pd.read_csv("disease_data.csv")

df.head(10).T
df.shape
df.nunique()
