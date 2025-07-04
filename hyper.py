from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data

df = load_data()
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define hyperparameter grids
param_grids = {
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0],
        "solver": ["auto", "svd", "cholesky"],
        "fit_intercept": [True, False]
    },
    "DecisionTree": {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["squared_error", "friedman_mse"]
    },
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }
}

models = {
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor()
}

# Run GridSearchCV for each model
for name in models:
    print(f"\nTuning {name}...")
    grid = GridSearchCV(models[name], param_grids[name], cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    print(f"{name} Best Params: {grid.best_params_}")
    print(f"{name} - MSE: {mean_squared_error(y_test, preds):.2f}, R²: {r2_score(y_test, preds):.2f}")
