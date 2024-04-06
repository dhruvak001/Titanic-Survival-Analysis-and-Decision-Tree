import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print(df)
# df.head()
# df.info()

# df = pd.DataFrame(url)

# Shuffle the DataFrame to randomize the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the size of each split
total_samples = len(df)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)

# Split the DataFrame into train, validation, and test sets
train_set = df.iloc[:train_size]
val_set = df.iloc[train_size: train_size + val_size]
test_set = df.iloc[train_size + val_size:]

# Print the lengths of the splits
print("Train set size:", len(train_set))
print("Validation set size:", len(val_set))
print("Test set size:", len(test_set))


import matplotlib.pyplot as plt

# Visualize the first few rows of the dataset
print(df.head())

# Visualize the survival distribution
plt.figure(figsize=(8, 6))
df['Survived'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Survival Distribution')
plt.xlabel('Survived (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()

import math

def entropy(survived_count, total_samples):
    p_survived = survived_count / total_samples
    p_not_survived = 1 - p_survived

    if p_survived == 0 or p_not_survived == 0:
        return 0  # To handle the case when log(0) occurs

    return - (p_survived * math.log2(p_survived) + p_not_survived * math.log2(p_not_survived))


# Example:
total_samples = len(train_set)
survived_df = df[(df['Survived'] == 1)]
print(survived_df)
survived_count = len(survived_df)
entropy_value = entropy(survived_count, total_samples)
print("Entropy of the training set:", entropy_value)

print(survived_df.head())

# Visualize the entropy of the training set
print("Entropy of the training set:", entropy_value)



def conTocat(feature_values, labels, criterion='mean'):
    if criterion == 'mean':
        threshold = sum(feature_values) / len(feature_values)
    elif criterion == 'median':
        sorted_values = sorted(feature_values)
        middle_index = len(sorted_values) // 2
        threshold = (sorted_values[middle_index - 1] + sorted_values[middle_index]) / 2

    categories = ['Category 1' if value <= threshold else 'Category 2' for value in feature_values]

    return categories

class DecisionNode:
    def __init__(self, feature, value, left, right):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self, labels):
        self.predicted_class = labels.mode().iloc[0]



def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def information_gain(parent_labels, left_labels, right_labels):
    parent_entropy = entropy(parent_labels)
    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)

    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)

    gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    return gain

def find_best_split(X, y):
    best_feature = None
    best_value = None
    best_gain = -1

    for feature in X.columns:
        values = X[feature].unique()

        for value in values:
            left_indices = X[feature] <= value
            right_indices = ~left_indices

            gain = information_gain(y, y[left_indices], y[right_indices])

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value

    return best_feature, best_value

def make_split(X, y, feature, value):
    left_indices = X[feature] <= value
    right_indices = ~left_indices

    left_X = X[left_indices]
    left_y = y[left_indices]

    right_X = X[right_indices]
    right_y = y[right_indices]

    return left_X, left_y, right_X, right_y

def train_tree(X, y, max_depth, current_depth=0):
    if current_depth == max_depth or not information_gain_possible(X, y):
        return LeafNode(y)

    best_feature, best_value = find_best_split(X, y)

    left_X, left_y, right_X, right_y = make_split(X, y, best_feature, best_value)

    left_subtree = train_tree(left_X, left_y, max_depth, current_depth + 1)
    right_subtree = train_tree(right_X, right_y, max_depth, current_depth + 1)

    return DecisionNode(best_feature, best_value, left_subtree, right_subtree)

def information_gain_possible(X, y):
    return len(set(y)) > 1

# Example usage:
max_depth = 5
trained_tree = train_tree(train_set.drop('Survived', axis=1), train_set['Survived'], max_depth)


def infer_tree(sample, tree):
    if isinstance(sample, pd.DataFrame):
        # If the input is a DataFrame, infer for each row
        predictions = [infer(row, tree) for _, row in sample.iterrows()]
        return np.array(predictions)
    else:
        # If the input is a single row, infer for that row
        return infer(sample, tree)

def infer(sample, tree):
    if isinstance(tree, DecisionNode):  # Checking if the node is a DecisionNode
        feature_value = sample[tree.feature]
        if pd.api.types.is_numeric_dtype(feature_value):
            # If the feature is numeric, check the split condition
            if feature_value <= tree.value:
                return infer(sample, tree.left)
            else:
                return infer(sample, tree.right)
        else:
            # If the feature is categorical, check the category
            if str(feature_value) == str(tree.value):
                return infer(sample, tree.left)
            else:
                return infer(sample, tree.right)
    else:
        # If the node is a LeafNode, return the predicted class
        return tree.predicted_class


# Example Usage:
test_predictions = infer_tree(test_set.drop('Survived', axis=1), trained_tree)
test_predictions_binary = np.array([infer(sample, trained_tree) for _, sample in test_set.drop('Survived', axis=1).iterrows()])


# Visualize the decision tree split
plt.figure(figsize=(12, 8))
# Assuming 'Age' is one of the features
plt.scatter(train_set['Age'], train_set['Survived'], c=train_set['Survived'], cmap='viridis', label='Survived')
plt.title('Decision Tree Split Visualization')
plt.xlabel('Age')
plt.ylabel('Survived (0: No, 1: Yes)')
plt.legend()
plt.show()

# Task 7
def compute_confusion_matrix(predictions, true_labels):
    cm = confusion_matrix(true_labels, predictions)
    return cm

# Example usage:
true_labels = test_set['Survived'].values
print(len(true_labels), len(test_predictions_binary))
conf_matrix = compute_confusion_matrix(test_predictions_binary, true_labels)

# Display the Confusion Matrix
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


from sklearn.metrics import precision_recall_fscore_support

# Task 8
def compute_precision_recall_f1(predictions, true_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    return precision, recall, f1

# Example usage:
precision, recall, f1 = compute_precision_recall_f1(test_predictions, true_labels)

# Display Precision, Recall, F1-score
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

