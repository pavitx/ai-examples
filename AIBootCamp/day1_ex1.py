import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

df = pd.read_csv(url)

features = df[['total_bill', 'size']]
target = df['tip']

print("Features: \n", features.head())
print("\nTarget: \n", target.head())

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("Training data set: ", X_train.shape)
print("Testing data set: ", X_test.shape)

sns.pairplot(df, x_vars=['total_bill', 'size'], y_vars=['tip'], height=5, aspect=0.8, kind="scatter")
plt.title("Feature vs target relationships")
plt.show()