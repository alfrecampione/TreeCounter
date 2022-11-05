from create_trains_test import *
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
print(
    my_dict[
        max(lin_rmse_scores.max(), max(tree_rmse_scores.max(), forest_rmse_score.max()))
    ]
)
