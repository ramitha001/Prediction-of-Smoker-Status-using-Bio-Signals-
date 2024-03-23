import pickle
import numpy as np

# Load the pickle file
with open(r"E:/COMPUTATIONAL INTELIGENCE/CSV Files/Assign_RF_model (79).pkf", "rb") as f:
    model = pickle.load(f)

# Identify and inspect the node array
node_array = model.tree_.__getstate__()['nodes']

# Check current dtype
print("Current dtype of node array:", node_array.dtype)

# Define the expected dtype
expected_dtype = np.dtype({'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
                                     'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'],
                           'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'],
                           'offsets': [0, 8, 16, 24, 32, 40, 48, 56], 'itemsize': 64})

# Check if dtype needs conversion
if node_array.dtype != expected_dtype:
    # Convert dtype
    converted_node_array = node_array.astype(expected_dtype)
    model.tree_.__setstate__({'nodes': converted_node_array})

    # Save the modified model to a new pickle file
    with open(r"E:/COMPUTATIONAL INTELIGENCE/CSV Files/Assign_RF_model_fixed.pkf", "wb") as f:
        pickle.dump(model, f)
    print("Model saved with fixed dtype.")
else:
    print("Node array dtype is already compatible.")

