import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # # Check command-line arguments
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(r"C:\Users\limji\Documents\Comp sci\Python\CS50_AI_course\shopping\shopping.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    month_map = {
    'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
    'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
}

    df = pd.read_csv(filename)
    df['Administrative'] = df["Administrative"].astype(int)
    df['Administrative_Duration'] = df["Administrative_Duration"].astype(float)
    df['Informational']  = df['Informational'].astype(int)
    df['Informational_Duration'] = df['Informational_Duration'].astype(float)
    df['ProductRelated'] = df['ProductRelated'].astype(int)
    df['ProductRelated_Duration'] = df['ProductRelated_Duration'].astype(float)
    df['BounceRates'] = df['BounceRates'].astype(float)
    df['ExitRates'] = df['ExitRates'] .astype(float)
    df['PageValues'] = df['PageValues'].astype(float)
    df['SpecialDay'] = df['SpecialDay'].astype(float)
    df['Month'] = df['Month'].map(month_map)
    df['OperatingSystems'] = df['OperatingSystems'].astype(int) 
    df['Browser'] = df['Browser'].astype(int) 
    df['Region'] = df['Region'].astype(int) 
    df['TrafficType'] = df['TrafficType'].astype(int) 
    df['VisitorType'] = df['VisitorType'].map(lambda x: 1 if x =="Returning_Visitor" else 0)
    df['Weekend'] = df['Weekend'].astype(int)
    df["Revenue"] = df['Revenue'].astype(int)
    
    evidence = df.iloc[:, :-1].values.tolist()
    label = df.iloc[:,-1].values.tolist()
    return (evidence, label)



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # model = Perceptron()
    model = svm.SVC()
    # model = KNeighborsClassifier(n_neighbors=5)
    # model = GaussianNB()

    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    n = len(labels)
    real_positive, pred_positive_acc = 0, 0
    real_negative, pred_negative_acc = 0, 0
    for i in range(n):
        if labels[i] == 1:
            real_positive += 1
            if predictions[i] == 1:
                pred_positive_acc += 1
        elif labels[i] == 0:
            real_negative += 1
            if predictions[i] == 0:
                pred_negative_acc += 1
    
    sensitivity = pred_positive_acc/real_positive
    specificity = pred_negative_acc/real_negative
    return (sensitivity, specificity)
        
if __name__ == "__main__":
    main()
