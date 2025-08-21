"""
Dataset configuration constants
"""

# Image scales for different datasets
TRAIN_SCALE = {
    'pascalcontext': (512, 512),
    'nyud': (480, 640),
}

TEST_SCALE = {
    'pascalcontext': (512, 512),
    'nyud': (480, 640),
}

# Number of training images
NUM_TRAIN_IMAGES = {
    'pascalcontext': 4998,
    'nyud': 795,
}

# Output channels for different tasks and datasets
def get_output_num(task, dataname):
    """Get number of output channels for task on dataset"""
    if dataname == 'pascalcontext':
        task_output = {
            'semseg': 21,
            'human_parts': 7,
            'normals': 3,
            'edge': 1,
            'sal': 2,
        }
    elif dataname == 'nyud':
        task_output = {
            'semseg': 40,
            'normals': 3,
            'edge': 1,
            'depth': 1,
        }
    else:
        raise NotImplementedError(f"Dataset {dataname} not supported")
    
    return task_output[task]


