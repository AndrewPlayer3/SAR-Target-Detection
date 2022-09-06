"""
Andrew Player
September 2022
Script for training and testing a network for MSTAR SAR Target Detection.
"""


import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mstar_io import test_with_val_data,  make_dataset
from training import train_model

DATA_DIR = "./data"
MODEL_NAME = f"Testing_{time.time()}"
EPOCHS = 10
BATCH_SIZE = 1
VALIDATION_SPLIT = 0.1


print("CREATING DATASET:")
print("-----------------")

"""
    Right now, the data dir should just contain the target folders, as such:
    ./data
        |_ BRDM_2/ 
        |_ .../
"""
save_directory, count, label_dict, target_types = make_dataset(DATA_DIR, validation_split=VALIDATION_SPLIT)

training_set_dir = "./" + str(save_directory)

print(f"Dataset saved to \"{training_set_dir}\"")
print(f"Dataset contains {count} samples.")
print(f"Labels to array index: {label_dict}\n")


print("TRAINING MODEL:")
print("--------------")

"""
    Basic conv to dense model for now.
"""
history = train_model(
    MODEL_NAME,
    training_set_dir, 
    128,
    EPOCHS,
    BATCH_SIZE
)


print("\nMODEL HISTORY:")
print("--------------")
print(history.history)
print("")


print("TEST RESULT:")
print("------------")

test_with_val_data(training_set_dir, "./models/checkpoints/"+MODEL_NAME)

print("")