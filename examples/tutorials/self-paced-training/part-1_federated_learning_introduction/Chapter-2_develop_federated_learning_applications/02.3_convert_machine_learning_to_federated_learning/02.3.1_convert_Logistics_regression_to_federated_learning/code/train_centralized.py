# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

DATA_ROOT = "/tmp/flare/dataset/heart_disease_data/"

MAX_ITERS = 4
EPSILON = 1.0


def sigmoid(inp):
    return 1.0 / (1.0 + np.exp(-inp))


def lr_solver(X, y):
    """
    Custom logistic regression solver using Newton Raphson
    method.

    """
    n_features = X.shape[1]
    theta = np.zeros((n_features + 1, 1))
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    for iter in range(MAX_ITERS):
        proba = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (y - proba))
        D = np.diag((proba * (1 - proba))[:, 0])
        hessian = X.T.dot(D).dot(X)

        reg = EPSILON * np.eye(hessian.shape[0])
        updates = np.linalg.solve(hessian + reg, gradient)

        theta += updates

    return theta


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=str,
        default="custom",
        help=("which solver to use: custom (default) or sklearn " "LogisticRegression. The results are the same. "),
    )
    args = parser.parse_args()

    print("using solver:", args.solver)

    print("loading training data.")
    train_X = np.concatenate(
        (
            np.load(os.path.join(DATA_ROOT, "site-1.train.x.npy")),
            np.load(os.path.join(DATA_ROOT, "site-2.train.x.npy")),
            np.load(os.path.join(DATA_ROOT, "site-3.train.x.npy")),
            np.load(os.path.join(DATA_ROOT, "site-4.train.x.npy")),
        )
    )
    train_y = np.concatenate(
        (
            np.load(os.path.join(DATA_ROOT, "site-1.train.y.npy")),
            np.load(os.path.join(DATA_ROOT, "site-2.train.y.npy")),
            np.load(os.path.join(DATA_ROOT, "site-3.train.y.npy")),
            np.load(os.path.join(DATA_ROOT, "site-4.train.y.npy")),
        )
    )

if args.solver == "sklearn":
    train_y = train_y.reshape(-1)

print("training data X loaded. shape:", train_X.shape)
print("training data y loaded. shape:", train_y.shape)

if args.solver == "sklearn":
    clf = LogisticRegression(random_state=0, solver="newton-cholesky", verbose=1).fit(train_X, train_y)

else:
    theta = lr_solver(train_X, train_y)

for site in range(4):

    print("\nsite - {}".format(site + 1))
    test_X = np.load(os.path.join(DATA_ROOT, "site-{}.test.x.npy".format(site + 1)))
    test_y = np.load(os.path.join(DATA_ROOT, "site-{}.test.y.npy".format(site + 1)))
    test_y = test_y.reshape(-1)

    print("validation set n_samples: ", test_X.shape[0])

    if args.solver == "sklearn":
        proba = clf.predict_proba(test_X)
        proba = proba[:, 1]

    else:
        test_X = np.concatenate((np.ones((test_X.shape[0], 1)), test_X), axis=1)
        proba = sigmoid(np.dot(test_X, theta))

    print("accuracy:", accuracy_score(test_y, proba.round()))
    print("precision:", precision_score(test_y, proba.round()))
