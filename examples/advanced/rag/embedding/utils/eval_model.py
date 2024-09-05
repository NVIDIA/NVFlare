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

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, TripletEvaluator


def main():
    # argparse
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument(
        "--model_path",
        type=str,
    )
    args = parser.parse_args()
    model_path = args.model_path

    # Load a model to finetune with
    model = SentenceTransformer(model_path)

    # Evaluate the trained model on the test set with embedding similarity
    dataset_test = load_dataset("sentence-transformers/stsb", split="validation")
    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=dataset_test["sentence1"],
        sentences2=dataset_test["sentence2"],
        scores=dataset_test["score"],
        main_similarity=SimilarityFunction.COSINE,
    )
    metric_score = test_evaluator(model)
    print(f"Test set evaluation on STSB: {metric_score}")

    # Evaluate the trained model on the test set with triplet loss
    dataset_test = load_dataset("sentence-transformers/all-nli", "triplet", split="test")
    test_evaluator = TripletEvaluator(
        anchors=dataset_test["anchor"],
        positives=dataset_test["positive"],
        negatives=dataset_test["negative"],
    )
    metric_score = test_evaluator(model)
    print(f"Test set evaluation on NLI: {metric_score}")


if __name__ == "__main__":
    main()
