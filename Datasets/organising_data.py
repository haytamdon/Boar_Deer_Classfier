import os
from sklearn.model_selection import train_test_split

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