from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# (1) import nvflare client API
from nvflare import client as flare


def main():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # (2) initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        global_params = input_model.params

        print(f"current_round={input_model.current_round}")

        model = None
        # Create a logistic regression model based on global model, since this is one-shot learning
        # model init parameters will be the same, Only difference model result (coef, intercept)
        model = LogisticRegression(random_state=42)

        # update model based on global parameters
        has_global_model = False
        if "coef" in global_params:
            model.coef_ = global_params["coef"]
        if model.fit_intercept and "intercept" in global_params:
            model.intercept_ = global_params["intercept"]
            has_global_model = True

        # evaluate global model first.
        if not has_global_model:
            # Train the model on the training set
            model.fit(X_train, y_train)

        accuracy, report = evaluate_model(X_test, model, y_test)

        # Print the results
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", report)

        #  construct trained FL model
        params = {"coef": model.coef_,"intercept": model.intercept_}
        metrics = {"accuracy": accuracy, "report": report}
        output_model = flare.FLModel(params=params,metrics = metrics)
        # send model back to NVFlare
        flare.send(output_model)


def evaluate_model(X_test, model, y_test):
    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


if __name__ == "__main__":
    main()
