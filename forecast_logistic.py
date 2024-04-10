import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
# Read the data
data = pd.read_csv('seattle-weather.csv')
data = data.set_index('date')

# Prepare the features and target
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']].values 
y = data['weather'].values 

# Function to split data into training and testing sets
def train_test_split(X, y, test_size, random_state=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


def compute_accuracy(y_test, y_pred):  
    matched = 0  
    for true_label, predicted in zip(y_test, y_pred):  
        if true_label == predicted:  
            matched += 1  
    accuracy_score = matched / len(y_pred)  
    return accuracy_score  

def classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)
    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total
    class_report_df['avg / total'] = avg
    return class_report_df.T

accuracy  = compute_accuracy(y_test,y_pred)
recall = classification_report(y_test,y_pred)

print(f"Accuracy: {accuracy:.2f}")

print(recall)