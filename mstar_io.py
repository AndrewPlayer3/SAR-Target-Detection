"""
Andrew Player
September 2022
Script for parsing and working with MSTAR files and the datasets.
"""

import os
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_mstar_file(
    filename : str
):

    """
    Parse the header and data from an MSTAR data file.

    Parameters:
    -----------
    filename : str
        The path to the *.0XX MSTAR file that should be read.

    Returns:
    --------
    magnitude : np.ndarray()
        The magnitude data from the MSTAR file. This is the image data.
    phase : np.ndarray()
        The phase data from the MSTAR file. (Not particullarly useful here.)
    header : dict
        The header information from the file. Contains target type, mission, resolution, etc...
    """    

    file = open(filename, "rb")
    file_contents = file.read()

    header_length = int(str(file_contents[46:51])[2:-1])

    header_info = file_contents[0:header_length]
    header_info = str(header_info)[2:-1].split('\\n')[2:-2]

    header = {}
    for info in header_info:
        spl  = info.split('= ')
        name = spl[0]
        val  = spl[1]
        header[name] = val

    num_cols = int(header['NumberOfColumns'])
    num_rows = int(header['NumberOfRows'])
    num_pxls = num_rows * num_cols

    format = ">" + str(num_pxls) + "f"

    data_raw_mag = file_contents[header_length:num_pxls*4 + header_length]
    data_raw_phs = file_contents[num_pxls*4 + header_length:]

    mag_data_unpacked = struct.unpack(format, data_raw_mag)
    phs_data_unpacked = struct.unpack(format, data_raw_phs)

    phase     = np.reshape(phs_data_unpacked, (num_rows, num_cols))
    magnitude = np.reshape(mag_data_unpacked, (num_rows, num_cols))

    return magnitude, phase, header


def get_files_and_types(
    data_dir: str
):
    
    """
    Gets the list of sample filepaths and the target_types.

    Parameters:
    -----------
    data_dir : str
        The directory which contains the data, before processing with make_dataset.

    Returns:
    --------
    data_files : list[str]
        The list of filepaths of data examples.
    target_types : list[str]
        The list of target types existing in the dataset.
    """

    data_files = []
    target_types = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename[-3:] != "JPG" and filename[-3:] != "HTM":

                filepath = os.path.join(root, filename)
                data_files.append(filepath)

                _, _, header = read_mstar_file(filepath)
                target_type = header['TargetType']
                if not target_type in target_types:
                    target_types.append(target_type)

    return data_files, target_types


def split_dataset(
    dataset_path: str,
    split:        float
):

    """
    Split the dataset into train and test folders

    Parameters:
    -----------
    dataset_path : str
        The path to the dataset to be split
    split : float
        The train/test split, 0 < Split < 1, size(validation) <= split

    Returns:
    --------
    num_train : int
        The number of elements that went to the training set.
    num_validation : int
        The number of elements that went to the validation set.
    """

    train_dir      = Path(dataset_path) / "train"
    validation_dir = Path(dataset_path) / "validation"

    try:
        train_dir.mkdir()
        validation_dir.mkdir()
    except OSError:
        print("\nTrain or Validation Dir already exists -- skipping.\n")

    num_train = 0
    num_validation = 0
    for _, _, filenames in os.walk(dataset_path):
        for filename in filenames:

            old_path = Path(dataset_path) / filename

            split_value = np.random.rand()
            if split_value <= split:
                num_validation += 1
                new_path = validation_dir / filename
            else:
                num_train += 1
                new_path = train_dir / filename

            try:
                os.rename(old_path, new_path)
            except OSError:
                pass
        break

    return num_train, num_validation


def make_dataset(
    in_data_dir:      str,
    out_data_dir:     str = ".",
    validation_split: int = 0.1
):
    
    """
    Create a ready-to-train dataset. 
    in_data_dir should just contain the folders with the raw and jpg files.

    Parameters:
    -----------
    in_data_dir : str
        -- BRDM_2
            -- *.0XX
            -- ...
        -- ...
    out_data_dir : str
        -- train
            -- *.0XX
        -- validation
            -- *.0XX
    split : float
        Decimal proportion that should go to validation. i.e. 0.1 == 10% 
    
    Returns:
    --------
    save_directory : str
        The name of the directory that was saved to. Does not include the root path.
    count : str
        The number of samples included in the dataset.
    labels : dict(str->int)
        Dictionary mapping target type to its index in the one-hot-encoded label array.
    target_types : list[str]
        List containing the found and included target types.
    """

    file_list, target_types = get_files_and_types(in_data_dir)

    index = 0
    labels = {}
    for target_type in target_types:
        labels[target_type] = index
        index += 1

    dir_name = f"MSTAR_DATASET_{time.time()}"

    save_directory = Path(out_data_dir) / dir_name
    if not save_directory.is_dir():
        save_directory.mkdir()

    count = 0
    for file in file_list:

        magnitude, _, header = read_mstar_file(file)
    
        num_rows = int(header['NumberOfRows'])
        num_cols = int(header['NumberOfColumns'])

        if num_rows >= 128 and num_cols >= 128:

            row_offset = (num_rows - 128) // 2
            col_offset = (num_cols - 128) // 2

            data = magnitude[row_offset:(128+row_offset), col_offset:(128+col_offset)]

            target_type = header['TargetType']
            out_name    = header['ParentScene'] + target_type

            label = np.zeros(len(target_types))
            label[labels[target_type]] = 1

            count += 1

            np.savez(
                os.path.join(save_directory, out_name),
                magnitude=data,
                label=label,
                header=header
            )

    split_dataset(save_directory, validation_split)

    return save_directory, count, labels, target_types


def plot_mstar_file(
    filename: str
):
    
    """
    Plot MSTAR sample via its filename
    """

    magnitude, _, _ = read_mstar_file(filename)

    plt.imshow(magnitude)
    plt.show()


def plot_mstar_sample(
    magnitude: str
):

    """
    Plot MSTAR sample via its magnitude data.
    """

    plt.imshow(magnitude)
    plt.show()


def load_sample(
    filename: str
):
    
    """
    Load magnitude and label information from a dataset file.
    """

    dataset_file = np.load(filename)
    return dataset_file['magnitude'], dataset_file['label']


def test_with_val_data(
    dataset_dir: str,
    model_path:  str
):

    """
    Run predictions on the validation samples of a dataset.
    """

    from tensorflow.keras.models import load_model

    model = load_model(model_path)

    val_path   = dataset_dir + '/validation'

    correct = 0
    incorrect = 0
    num_samples = 0

    for filename in os.listdir(val_path):

        val_file = os.path.join(val_path, filename)

        magnitude, label_true = load_sample(val_file)

        label = model.predict(magnitude.reshape((1, 128, 128, 1)))

        target_true = np.argmax(label_true)
        target_pred = np.argmax(label)

        if target_true == target_pred:
            correct += 1
        else:
            incorrect += 1

        num_samples += 1 

    print(f"Samples   {num_samples}")
    print(f"Correct   {correct}")
    print(f"Incorrect {incorrect}")
    print(f"Accuracy  {100 * (correct / num_samples)}%")


