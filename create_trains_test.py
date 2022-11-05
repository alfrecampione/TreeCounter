import os
import pandas as pd


def load_database(database_path: str, file_name: str):
    csv_path = os.path.join(database_path, file_name)
    return pd.read_csv(csv_path)


database = load_database("", "database.csv")

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(database, database["tree_count"]):
    strat_train_set = database.loc[train_index]
    strat_test_set = database.loc[test_index]
# database_label = strat_train_set["tree_count"].copy()
# for set_ in (strat_train_set, strat_test_set):
#    set_.drop("tree_count", axis=1, inplace=True)
# database = set_.copy()
