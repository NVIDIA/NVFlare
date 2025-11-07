# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


"""
Basic Usage Example for Feature Election in NVIDIA FLARE

This example demonstrates the simplest way to use Feature Election
for federated feature selection on tabular datasets.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from nvflare.app_opt.feature_election import quick_election


def create_sample_dataset():
    """Create a sample high-dimensional dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_redundant=30,
        n_repeated=10,
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [f"feature_{i:03d}" for i in range(100)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Created dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    return df


def example_1_quick_start():
    """Example 1: Quickstart - simplest usage"""
    print("\n" + "="*60)
    print("Example 1: Quick Start")
    print("="*60)
    
    # Create dataset
    df = create_sample_dataset()
    
    # Run Feature Election with just one line!
    selected_mask, stats = quick_election(
        df=df,
        target_col='target',
        num_clients=4,
        fs_method='lasso',
        auto_tune=True
    )
    
    # Print results
    print(f"\nOriginal features: {stats['num_features_original']}")
    print(f"Selected features: {stats['num_features_selected']}")
    print(f"Reduction: {stats['reduction_ratio']:.1%}")
    print(f"Optimal freedom_degree: {stats['freedom_degree']:.2f}")
    
    # Get selected feature names
    feature_names = [col for col in df.columns if col != 'target']
    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    print(f"\nFirst 10 selected features: {selected_features[:10]}")


def example_2_with_evaluation():
    """Example 2: With model evaluation"""
    print("\n" + "="*60)
    print("Example 2: With Model Evaluation")
    print("="*60)
    
    # Create dataset
    df = create_sample_dataset()
    
    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Prepare DataFrame for feature election (using training data only)
    df_train = X_train.copy()
    df_train['target'] = y_train
    
    # Run Feature Election
    selected_mask, stats = quick_election(
        df=df_train,
        target_col='target',
        num_clients=4,
        fs_method='lasso',
        auto_tune=True
    )
    
    # Apply mask to get selected features
    X_train_selected = X_train.iloc[:, selected_mask]
    X_test_selected = X_test.iloc[:, selected_mask]
    
    # Train models
    print("\nTraining models...")
    
    # Model with all features
    clf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_all.fit(X_train, y_train)
    y_pred_all = clf_all.predict(X_test)
    
    # Model with selected features
    clf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_selected.fit(X_train_selected, y_train)
    y_pred_selected = clf_selected.predict(X_test_selected)
    
    # Compare results
    print("\nResults:")
    print("-" * 60)
    print(f"{'Metric':<20} {'All Features':<20} {'Selected Features':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {accuracy_score(y_test, y_pred_all):<20.4f} {accuracy_score(y_test, y_pred_selected):<20.4f}")
    print(f"{'F1 Score':<20} {f1_score(y_test, y_pred_all):<20.4f} {f1_score(y_test, y_pred_selected):<20.4f}")
    print(f"{'# Features':<20} {X_train.shape[1]:<20} {X_train_selected.shape[1]:<20}")
    print("-" * 60)


def example_3_custom_configuration():
    """Example 3: Custom configuration"""
    print("\n" + "="*60)
    print("Example 3: Custom Configuration")
    print("="*60)
    
    from nvflare.app_opt.feature_election import FeatureElection
    
    # Create dataset
    df = create_sample_dataset()
    
    # Initialize with custom parameters
    fe = FeatureElection(
        freedom_degree=0.6,
        fs_method='elastic_net',
        aggregation_mode='weighted'
    )
    
    # Prepare data splits
    client_data = fe.prepare_data_splits(
        df=df,
        target_col='target',
        num_clients=5,
        split_strategy='stratified'
    )
    
    print(f"Prepared data for {len(client_data)} clients")
    for i, (X, y) in enumerate(client_data):
        print(f"  Client {i+1}: {len(X)} samples, class distribution: {y.value_counts().to_dict()}")
    
    # Run election
    stats = fe.simulate_election(client_data)
    
    # Print results
    print(f"\nElection Results:")
    print(f"  Features selected: {stats['num_features_selected']}/{stats['num_features_original']}")
    print(f"  Reduction: {stats['reduction_ratio']:.1%}")
    print(f"  Intersection features: {stats['intersection_features']}")
    print(f"  Union features: {stats['union_features']}")
    
    # Print client statistics
    print(f"\nPer-Client Statistics:")
    for client_name, client_stats in stats['client_stats'].items():
        print(f"  {client_name}:")
        print(f"    Features selected: {client_stats['num_selected']}")
        print(f"    Score improvement: {client_stats['improvement']:+.4f}")
    
    # Save results
    fe.save_results("feature_election_results.json")
    print("\n✓ Results saved to feature_election_results.json")


def example_4_different_methods():
    """Example 4: Compare different feature selection methods"""
    print("\n" + "="*60)
    print("Example 4: Comparing Different FS Methods")
    print("="*60)
    
    # Create dataset
    df = create_sample_dataset()
    
    methods = ['lasso', 'elastic_net', 'random_forest', 'mutual_info', 'f_classif']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method}...")
        selected_mask, stats = quick_election(
            df=df,
            target_col='target',
            num_clients=4,
            fs_method=method,
            auto_tune=False,
            freedom_degree=0.5
        )
        
        results[method] = {
            'selected': stats['num_features_selected'],
            'reduction': stats['reduction_ratio'],
            'intersection': stats['intersection_features'],
            'union': stats['union_features']
        }
    
    # Display comparison
    print("\n" + "="*60)
    print("Method Comparison")
    print("="*60)
    print(f"{'Method':<15} {'Selected':<12} {'Reduction':<12} {'Intersection':<12} {'Union':<10}")
    print("-" * 60)
    for method, res in results.items():
        print(f"{method:<15} {res['selected']:<12} {res['reduction']:<11.1%} {res['intersection']:<12} {res['union']:<10}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" Feature Election for NVIDIA FLARE - Basic Examples")
    print("="*70)
    
    try:
        example_1_quick_start()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_with_evaluation()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_custom_configuration()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_different_methods()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    print("\n" + "="*70)
    print(" All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
