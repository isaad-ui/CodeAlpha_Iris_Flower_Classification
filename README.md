# CodeAlpha_Iris_Flower_Classification

## Overview
This project was completed as part of the Data Science Internship at CodeAlpha. The goal is to build a machine learning model that classifies iris flowers into one of three species — setosa, versicolor, or virginica — based on four measurements: sepal length, sepal width, petal length, and petal width.

## Dataset
The dataset used is `Iris.csv`, containing 150 samples (50 for each species) with the following columns:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

## Approach
1. Loaded and explored the dataset using pandas.
2. Visualized relationships between measurements using pairplots and boxplots to see which features best separate the species.
3. Split the data into training (80%) and testing (20%) sets.
4. Trained a Decision Tree Classifier using scikit-learn.
5. Evaluated the model on the test set using accuracy score, classification report, and a confusion matrix.

## Results
The model achieved **100% accuracy** on the test set. This is expected for the Iris dataset since the three species are well separated by petal measurements, making this a relatively simple classification problem.

## Tools Used
- Python
- pandas
- matplotlib / seaborn
- scikit-learn

## Files in this Repository
- `iris_classification.py` — main script (data loading, EDA, model training, evaluation)
- `Iris.csv` — dataset used
- `iris_pairplot.png` — pairplot visualization
- `petal_length_boxplot.png` — boxplot of petal length by species
- `README.md` — this file

## How to Run
1. Make sure `Iris.csv` is in the same folder as `iris_classification.py`.
2. Install the required libraries: `pip install pandas matplotlib seaborn scikit-learn`
3. Run the script: `python iris_classification.py`
