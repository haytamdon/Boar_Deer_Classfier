import os
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd

def create_folders(image_types: list,
                    train_dir_path: str,
                    test_dir_path: str) -> None:
    """
    We create we folder to match the format indicated previously

    Args:
        image_types (List[str]): list of the image classes
        train_dir_path (str): path to the train directory
        test_dir_path (str): path to the test directory
    """
    os.makedirs(train_dir_path, exist_ok=True) # Creates folder if it doesn't exist
    os.makedirs(test_dir_path, exist_ok=True)
    for classe in image_types:
        # We create the paths to the subfolders for each class
        train_class_dir_path = os.path.join(train_dir_path, classe)
        test_class_dir_path = os.path.join(test_dir_path, classe)
        os.makedirs(train_class_dir_path, exist_ok=True)
        os.makedirs(test_class_dir_path, exist_ok=True)
    
def split_train_test_images(image_data):
    """
    Splits the data into training and testing set

    Args:
        image_data (pd.core.frame.DataFrame): df of all the image names

    Returns:
        Tuple(pd.core.frame.DataFrame): train & test dataframes
    """
    df_train, df_test = train_test_split(image_data,
                                        test_size=0.2, # 20% of the data will be in the test set
                                        stratify=image_data[["image_class"]], # we stratify the split on the class
                                        random_state=42) # To ensure we get the same random split everytime
    return df_train, df_test

def copy_images_to_appropriate_folder(df_train:pd.core.frame.DataFrame,
                                    df_test: pd.core.frame.DataFrame,
                                    train_dir_path: str,
                                    test_dir_path: str,
                                    image_types) -> None:
    """
    Copy the images to their appropriate folder in the previously mentioned
    structure according to their class and which set (trainset, testset)
    they belong to according to the previous split

    Args:
        df_train (pd.core.frame.DataFrame): train dataframe
        df_test (pd.core.frame.DataFrame): test dataframe
        train_dir_path (str): training data directory
        test_dir_path (str): testing data directory
        image_types (List[str]): list of the image classes
    """
    for classe in image_types:
        train_class_dir_path = os.path.join(train_dir_path, classe)
        test_class_dir_path = os.path.join(test_dir_path, classe)
        train_subset = df_train.loc[df_train["image_class"]==classe]
        test_subset = df_test.loc[df_test["image_class"]==classe]
        for i in range(len(train_subset)):
            org_file_path = train_subset.iloc[[i]]["image_path"].values[0]
            shutil.copy(org_file_path, train_class_dir_path)
        for i in range(len(test_subset)):
            org_file_path = test_subset.iloc[[i]]["image_path"].values[0]
            shutil.copy(org_file_path, test_class_dir_path)