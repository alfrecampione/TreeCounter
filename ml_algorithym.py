from create_trains_test import *
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

database = strat_train_set.drop("tree_count", axis=1, inplace=False)
values = strat_train_set["tree_count"]
lin_reg = LinearRegression()
lin_reg.fit(database, values)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(database, values)
forest_reg = RandomForestRegressor()
forest_reg.fit(database, values)


from sklearn.model_selection import cross_val_score

tree_scores = cross_val_score(
    tree_reg, database, values, scoring="neg_mean_squared_error"
)
tree_rmse_scores = np.sqrt(-tree_scores)

lin_scores = cross_val_score(
    lin_reg, database, values, scoring="neg_mean_squared_error"
)
lin_rmse_scores = np.sqrt(-lin_scores)

forest_score = cross_val_score(
    forest_reg,
    database,
    values,
    scoring="neg_mean_squared_error",
)
forest_rmse_score = np.sqrt(-forest_score)
my_dict = {
    lin_rmse_scores.max(): "linear",
    tree_rmse_scores.max(): "tree",
    forest_rmse_score.max(): "forest",
}
best_result = min(
    lin_rmse_scores.max(), min(tree_rmse_scores.max(), forest_rmse_score.max())
)

print(my_dict[best_result])

# model selection
best_model = {
    lin_rmse_scores.max(): lin_reg,
    tree_rmse_scores.max(): tree_reg,
    forest_rmse_score.max(): forest_reg,
}
final_model = best_model[best_result]

# Evaluate in test set
X_test = strat_test_set.drop("tree_count", axis=1)
Y_test = strat_test_set["tree_count"]
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_mse)
