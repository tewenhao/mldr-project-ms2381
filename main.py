from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

def prep_dataset(filename):
    """
    --- data cleaning for this specific dataset ---

    pd.to_numeric: The original TotalCharges column was being read as text because of empty spaces. 
    This converts it to numbers so the model can calculate trends.

    iloc[:, 1:20]: Your guide used 3:13, but your data has more features.
    We use 1:20 to include everything from gender up to TotalCharges.

    pd.get_dummies: Since you have many text categories (like InternetService), this step converts
    them into separate "0 or 1" columns. XGBoost cannot process the word "Fiber optic" directly,
    but it can process a "1" in an InternetService_Fiber optic column.

    Target Encoding: We converted "Yes/No" in $y$ to $1/0$ because Scikit-Learn's XGBoost classifier expects binary integers.
    """
    # import dataset
    dataset = pd.read_csv(filename)

    # data cleaning - clean TotalCharges column
    # contains spaces, so we convert it to numeric and fill missing values with 0
    dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce").fillna(0)

    # select features X and target Y
    # skip customerID (index 0) and take the rest of the indices
    # last column is churn
    X = dataset.iloc[:, 1:20]
    y = dataset.iloc[:, 20].values

    # map target churn to numbers since xgboost requires 0 and 1
    y = np.where(y == "Yes", 1, 0)

    # one hot encode categorical features in X
    X = pd.get_dummies(X, drop_first=True)

    print(f"X shape: {X.shape}")

    # split dataset into train/test
    # random_state seed for reproducable results
    # test size can be changed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test

def xgboost(X_train, X_test, y_train, y_test):
    # convert dataset into DMatrix
    xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # create xgboost model
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
    }
    n=50
    model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)

    # make predictions
    preds = model.predict(xgb_test)
    preds = np.round(preds)
    accuracy = accuracy_score(y_test,preds)
    print('Accuracy of XGBoost is:', accuracy*100)

    # see feature importance
    xgb.plot_importance(model)
    plt.show()

    return model

def lin_reg(X_train, X_test, y_train, y_test):
    # standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train linear regression model by fitting model to training data
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # make predictions
    y_pred = model.predict(X_test_scaled)

    # measure performance

    # Round the decimals to the nearest integer (0 or 1)
    y_pred_binary = np.round(y_pred).astype(int)
    # Optional: Clip values to ensure they stay within [0, 1] range
    y_pred_binary = np.clip(y_pred_binary, 0, 1)

    # Now accuracy_score will work
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy of linear regression is: {accuracy * 100}")

    # Calculate and print R^2 score.
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2:.4f}")

    # Calculate and print MSE.
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: {mse:.4f}")

    # Calculate and print RMSE.
    rmse = mse ** 0.5
    print(f"Root mean squared error: {rmse:.4f}")

    return model


def main():
    X_train, X_test, y_train, y_test = prep_dataset("data/data.csv")
    xgboost_model = xgboost(X_train, X_test, y_train, y_test)
    lin_reg_model = lin_reg(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()