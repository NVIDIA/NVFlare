"""
Production FLARE Deployment Example

This example shows how to deploy Feature Election in a real NVIDIA FLARE environment
with multiple clients, proper job configuration, and result collection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from nvflare.app_opt.feature_election import FeatureElection, FeatureElectionExecutor


def example_server_setup():
    """
    Server-side: Generate FLARE job configuration
    Run this on the server/admin machine
    """
    print("\n" + "="*70)
    print("SERVER SETUP: Creating FLARE Job Configuration")
    print("="*70)
    
    # Initialize Feature Election with your parameters
    fe = FeatureElection(
        freedom_degree=0.5,  # Will select features between intersection and union
        fs_method='lasso',   # Feature selection method
        aggregation_mode='weighted'  # Weight by sample count
    )
    
    # Generate FLARE job configuration
    job_paths = fe.create_flare_job(
        job_name="healthcare_feature_selection",
        output_dir="./flare_jobs",
        min_clients=3,  # Minimum 3 hospitals must participate
        num_rounds=1,   # Single round for feature selection
        client_sites=['hospital_a', 'hospital_b', 'hospital_c', 'hospital_d']
    )
    
    print("\n✓ Job configuration created:")
    print(f"  Job directory: {job_paths['job_dir']}")
    print(f"  Server config: {job_paths['server_config']}")
    print(f"  Client config: {job_paths['client_config']}")
    print(f"  Meta config: {job_paths['meta']}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the generated configuration files")
    print("2. Customize if needed (e.g., add privacy filters)")
    print("3. Each client should run the client_setup() function")
    print("4. Submit the job:")
    print(f"   nvflare job submit -j {job_paths['job_dir']}")
    print("="*70)
    
    return job_paths


def example_client_setup():
    """
    Client-side: Prepare and load data for Feature Election
    Run this on each client machine
    """
    print("\n" + "="*70)
    print("CLIENT SETUP: Preparing Data for Feature Election")
    print("="*70)
    
    # Simulate loading client's private data
    # In production, this would load from your actual data source
    print("\nLoading client data...")
    X_train, y_train, feature_names = load_client_data()
    
    print(f"  Loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Class distribution: {np.bincount(y_train.astype(int))}")
    
    # Initialize the executor
    executor = FeatureElectionExecutor(
        fs_method='lasso',
        eval_metric='f1',
        quick_eval=True
    )
    
    # Set the client's data
    executor.set_data(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names
    )
    
    print("\n✓ Client executor configured and ready")
    print("\nClient is now ready to participate in feature election")
    print("Wait for the server to submit the job...")
    
    return executor


def load_client_data():
    """
    Simulate loading client data
    In production, replace this with your actual data loading logic
    """
    from sklearn.datasets import make_classification
    
    # Simulate client-specific data
    X, y = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=20,
        n_redundant=30,
        random_state=np.random.randint(0, 1000)  # Each client has different data
    )
    
    feature_names = [f"biomarker_{i:03d}" for i in range(50)] + \
                   [f"clinical_{i:03d}" for i in range(30)] + \
                   [f"imaging_{i:03d}" for i in range(20)]
    
    return X, y, feature_names


def example_retrieve_results():
    """
    After job completion: Retrieve and analyze results
    Run this on the server/admin machine
    """
    print("\n" + "="*70)
    print("RETRIEVING RESULTS: After Job Completion")
    print("="*70)
    
    # In production, you would use FLARE API to get results
    # For this example, we'll simulate loading from a results file
    
    print("\nRetrieving results from FLARE server...")
    
    # Simulated result retrieval
    # In production:
    # from nvflare.fuel.flare_api.flare_api import new_secure_session
    # session = new_secure_session()
    # job_result = session.get_job_result(job_id)
    # global_mask = job_result['global_feature_mask']
    
    # For this example, we'll simulate with saved results
    from nvflare.app_opt.feature_election import load_election_results
    
    try:
        results = load_election_results("feature_election_results.json")
        
        print("\n✓ Results retrieved successfully")
        print(f"\nFeature Selection Summary:")
        print(f"  Original features: {results['election_stats']['num_features_original']}")
        print(f"  Selected features: {results['election_stats']['num_features_selected']}")
        print(f"  Reduction ratio: {results['election_stats']['reduction_ratio']:.1%}")
        print(f"  Freedom degree used: {results['freedom_degree']:.2f}")
        
        # Get selected feature names
        selected_features = results['selected_feature_names']
        print(f"\n  Selected feature names: {selected_features[:10]}...")
        
        # Client statistics
        print(f"\nPer-Client Statistics:")
        for client_name, client_stats in results['election_stats']['client_stats'].items():
            print(f"  {client_name}:")
            print(f"    Features selected: {client_stats['num_selected']}")
            print(f"    Performance improvement: {client_stats['improvement']:+.4f}")
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Apply the global feature mask to your datasets")
        print("2. Retrain models using only selected features")
        print("3. Evaluate performance improvement")
        print("4. Optional: Run federated learning with reduced features")
        print("="*70)
        
    except FileNotFoundError:
        print("\nNo results file found. Simulating results...")
        print("In production, results would be retrieved from FLARE server")


def example_apply_mask_to_new_data():
    """
    Apply the learned feature mask to new data
    """
    print("\n" + "="*70)
    print("APPLYING MASK: Using Selected Features on New Data")
    print("="*70)
    
    # Load the election results
    from nvflare.app_opt.feature_election import load_election_results
    
    try:
        results = load_election_results("feature_election_results.json")
        global_mask = np.array(results['global_mask'])
        
        # Simulate loading new data
        print("\nLoading new data for inference...")
        from sklearn.datasets import make_classification
        X_new, y_new = make_classification(
            n_samples=200,
            n_features=len(global_mask),
            random_state=42
        )
        
        print(f"  New data: {X_new.shape[0]} samples, {X_new.shape[1]} features")
        
        # Apply the mask
        X_new_selected = X_new[:, global_mask]
        
        print(f"  After selection: {X_new_selected.shape[0]} samples, {X_new_selected.shape[1]} features")
        print(f"  Reduction: {(1 - X_new_selected.shape[1]/X_new.shape[1]):.1%}")
        
        # Now use X_new_selected for training/inference
        print("\n✓ Feature mask successfully applied to new data")
        print("  Ready for model training or inference")
        
    except FileNotFoundError:
        print("\nNo results file found. Run the feature election first.")


def example_complete_workflow():
    """
    Complete workflow from setup to deployment
    """
    print("\n" + "="*70)
    print("COMPLETE WORKFLOW: End-to-End Feature Election")
    print("="*70)
    
    print("\n" + "-"*70)
    print("STEP 1: Server Setup")
    print("-"*70)
    job_paths = example_server_setup()
    
    print("\n" + "-"*70)
    print("STEP 2: Client Setup (run on each client)")
    print("-"*70)
    print("\nSimulating 3 clients...")
    for i in range(3):
        print(f"\n--- Client {i+1} ---")
        executor = example_client_setup()
    
    print("\n" + "-"*70)
    print("STEP 3: Job Execution")
    print("-"*70)
    print("\nIn production, the FLARE server would now:")
    print("1. Distribute the feature election task to all clients")
    print("2. Collect feature selections from each client")
    print("3. Aggregate selections using the specified freedom_degree")
    print("4. Distribute the global feature mask back to clients")
    
    print("\n" + "-"*70)
    print("STEP 4: Retrieve and Apply Results")
    print("-"*70)
    example_retrieve_results()
    example_apply_mask_to_new_data()


def example_with_privacy_filters():
    """
    Example with differential privacy filters (advanced)
    """
    print("\n" + "="*70)
    print("ADVANCED: Feature Election with Privacy Filters")
    print("="*70)
    
    print("\nTo add differential privacy to feature selection:")
    print("\n1. Modify the client config to include privacy filters:")
    print("""
    {
        "task_result_filters": [
            {
                "tasks": ["feature_election"],
                "filters": [
                    {
                        "name": "DPFilter",
                        "args": {
                            "epsilon": 1.0,
                            "noise_type": "gaussian"
                        }
                    }
                ]
            }
        ]
    }
    """)
    
    print("\n2. This will add noise to feature scores before sharing")
    print("3. Adjust epsilon based on your privacy requirements")
    print("   - Lower epsilon = more privacy, less accuracy")
    print("   - Higher epsilon = less privacy, more accuracy")


def main():
    """Run deployment examples"""
    print("\n" + "="*70)
    print(" Feature Election - Production FLARE Deployment Guide")
    print("="*70)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "server":
            example_server_setup()
        elif command == "client":
            example_client_setup()
        elif command == "results":
            example_retrieve_results()
        elif command == "apply":
            example_apply_mask_to_new_data()
        elif command == "privacy":
            example_with_privacy_filters()
        else:
            print(f"Unknown command: {command}")
            print_usage()
    else:
        # Run complete workflow
        example_complete_workflow()


def print_usage():
    """Print usage instructions"""
    print("\nUsage:")
    print("  python flare_deployment.py              # Run complete workflow")
    print("  python flare_deployment.py server       # Server setup only")
    print("  python flare_deployment.py client       # Client setup only")
    print("  python flare_deployment.py results      # Retrieve results")
    print("  python flare_deployment.py apply        # Apply mask to new data")
    print("  python flare_deployment.py privacy      # Privacy filters info")


if __name__ == "__main__":
    main()
