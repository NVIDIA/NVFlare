# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import runpy
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent


class SeededServerRun(FLComponent):
    """Make KMeansAssembler's unspecified server RNG repeatable for comparison."""

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            np.random.seed(731)


def classifier_state(model):
    return {
        "parameters": model.get_params(deep=False),
        **{
            name: getattr(model, name)
            for name in ("coef_", "intercept_", "classes_", "n_features_in_", "n_iter_", "t_")
        },
    }


def main():
    import nvflare.client as flare

    original_receive, original_send = flare.receive, flare.send
    original_fit = SGDClassifier.fit
    folder = Path("sklearn_evidence")
    folder.mkdir(exist_ok=True)
    current_round = None

    def receive(*args, **kwargs):
        nonlocal current_round
        model = original_receive(*args, **kwargs)
        current_round = model.current_round
        joblib.dump({"params": model.params, "meta": model.meta}, folder / f"{current_round}_input.joblib")
        return model

    def send(model, *args, **kwargs):
        joblib.dump(
            {"params": model.params, "metrics": model.metrics, "meta": model.meta, "pid": os.getpid()},
            folder / f"{current_round}_output.joblib",
        )
        return original_send(model, *args, **kwargs)

    def fit(model, *args, **kwargs):
        result = original_fit(model, *args, **kwargs)
        joblib.dump(classifier_state(model), folder / f"{current_round}_classifier.joblib")
        return result

    flare.receive, flare.send = receive, send
    SGDClassifier.fit = fit
    runpy.run_path(str(Path(__file__).with_name("shipped_client.py")), run_name="__main__")


if __name__ == "__main__":
    main()
