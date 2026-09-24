"""
CodeAlpha Data Science Internship - Task 1
Iris Flower Classification

What this project does (in plain terms):
We have measurements (like height and width of petals/sepals) for 150 iris
flowers, and each flower belongs to one of 3 species: setosa, versicolor,
or virginica. We'll teach a machine learning model to look at a flower's
measurements and guess which species it is - kind of like teaching someone
to identify dog breeds just from paw size and ear length.

Dataset: Iris.csv (the file provided for this task)
"""

# ---------- STEP 1: Import the tools we need ----------
import pandas as pd                      # for working with data in table form
import matplotlib.pyplot as plt          # for making charts
import seaborn as sns                    # for nicer-looking charts
from sklearn.model_selection import train_test_split   # to split data into train/test
from sklearn.tree import DecisionTreeClassifier          # our ML model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- STEP 2: Load the dataset ----------
# Change this path if your CSV is somewhere else (e.g. if you uploaded it
# to Colab, it'll just be "Iris.csv" if it's in the same folder as this script)
df = pd.read_csv("Iris.csv")

# The "Id" column is just a row number, not useful for prediction, so drop it
df = df.drop(columns=["Id"])

# ---------- STEP 3: Take a first look at the data ----------
print("First 5 rows of the dataset:")
print(df.head())

print("\nHow many flowers of each species do we have?")
print(df['Species'].value_counts())

print("\nBasic statistics (min, max, average, etc.) for each measurement:")
print(df.describe())

# ---------- STEP 4: Explore the data visually (EDA) ----------
# A pairplot shows every measurement plotted against every other measurement,
# with each species colored differently. This helps us SEE which measurements
# are good at telling the species apart before we even build a model.
sns.pairplot(df, hue='Species')
plt.suptitle("Iris Measurements by Species", y=1.02)
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved chart as iris_pairplot.png")

# A boxplot for petal length specifically - petals tend to be the strongest
# clue for telling species apart.
plt.figure(figsize=(8, 5))
sns.boxplot(x='Species', y='PetalLengthCm', data=df)
plt.title("Petal Length by Species")
plt.savefig("petal_length_boxplot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved chart as petal_length_boxplot.png")

# ---------- STEP 5: Split into training data and testing data ----------
# X = the measurements (inputs). y = the species (what we want to predict).
feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_columns]
y = df["Species"]

# We hold back 20% of the flowers as a "test" - like practice exam questions
# the model has never seen, so we can honestly check how well it learned.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining on {len(X_train)} flowers, testing on {len(X_test)} flowers.")

# ---------- STEP 6: Train the model ----------
# A Decision Tree works like a flowchart of yes/no questions
# (e.g. "Is petal length > 2.5cm? If yes, go this way...") until it lands
# on a species guess.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel trained!")

# ---------- STEP 7: Test the model and check accuracy ----------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")

print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix (rows = actual species, columns = predicted species):")
print(confusion_matrix(y_test, predictions))
