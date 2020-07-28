from sklearn.model_selection import train_test_split
import numpy as np
"""
    prepare data
"""


def load_random_data():
    # local generate
    origin_metric = np.random.rand(30, 4)
    x = origin_metric[:, :-1]
    y = origin_metric[:, -1].reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    id = np.array([id for id in range(1, x_test.shape[0]+1)], dtype=int).reshape(-1,1)
    return id, x_train, x_test, y_train, y_test
