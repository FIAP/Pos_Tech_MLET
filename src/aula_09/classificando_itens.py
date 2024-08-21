from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, fbeta_score, recall_score, precision_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def classify_items(consumption_list):
    # Split the consumption list into features (X) and labels (y)
    X = [consumption[:-1] for consumption in consumption_list]
    y = [consumption[-1] for consumption in consumption_list]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the classifiers for the voting ensemble
    classifiers = [
        ('decision_tree', DecisionTreeClassifier()),
        ('knn', KNeighborsClassifier()),
        ('logistic_regression', LogisticRegression())
    ]

    # Create a pipeline with MinMaxScaler and VotingClassifier
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('voting_classifier', VotingClassifier(classifiers))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    f_beta_score = fbeta_score(y_test, y_pred, beta=1)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc = roc_curve(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"F-beta Score: {f_beta_score}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"ROC Curve: {roc}")

    return accuracy, roc
