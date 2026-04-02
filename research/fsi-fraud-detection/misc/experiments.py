# Data paths for experiments (root relative to this file)
import os

from pathlib import Path
_src_misc_root = Path(__file__).resolve().parent.parent
sim_data_root = str(_src_misc_root / "datasets")

data_paths = {
    #### SIMULATION #####
    "sim-exp": {
        "siteA": {
            "data_root": os.path.join(sim_data_root, "siteA"),
            "train_data_path": "siteA_[type1_type2_type3]_[app_frac_0.9]_[pct_overlap_10]_[train_1].csv",
            "test_data_path": "siteA_*eval_1*.csv",
            "scaling_data_path": os.path.join(sim_data_root, "universal_scaling_datasets_all_banks.csv"),
        },
        "siteB": {
            "data_root": os.path.join(sim_data_root, "siteB"),
            "train_data_path": "siteB_[type2_type3]_[app_frac_0.9]_[pct_overlap_10]_[train_1].csv",
            "test_data_path": "siteB_*eval_1*.csv",
            "scaling_data_path": os.path.join(sim_data_root, "universal_scaling_datasets_all_banks.csv"),
        },
        "siteC": {  
            "data_root": os.path.join(sim_data_root, "siteC"),
            "train_data_path": "siteC_[type4_type3]_[app_frac_0.9]_[pct_overlap_10]_[train_1].csv",
            "test_data_path": "siteC_*eval_1*.csv",
            "scaling_data_path": os.path.join(sim_data_root, "universal_scaling_datasets_all_banks.csv"),
        },
        "siteD": {
            "data_root": os.path.join(sim_data_root, "siteD"),
            "train_data_path": "siteD_[type1_type4]_[app_frac_0.9]_[pct_overlap_10]_[train_1].csv",
            "test_data_path": "siteD_*eval_1*.csv",
            "scaling_data_path": os.path.join(sim_data_root, "universal_scaling_datasets_all_banks.csv"),
        },
        "siteE": {
            "data_root": os.path.join(sim_data_root, "siteE"),
            "train_data_path": "siteE_[type1_type2]_[app_frac_0.9]_[pct_overlap_10]_[train_1].csv",
            "test_data_path": "siteE_*eval_1*.csv",
            "scaling_data_path": os.path.join(sim_data_root, "universal_scaling_datasets_all_banks.csv"),
        },
    },     

    ### CENTRAL TRAINING ####
    "central-exp": {
        "central": {  
            "data_root": sim_data_root,
            "train_data_path": "*/*pct_overlap_10*train_1*csv",
            "test_data_path": "*/*eval_1*.csv",
        }
    }        
}
