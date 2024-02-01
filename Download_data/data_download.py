import requests
import os
from typing import List
import urllib
import wget
import pandas as pd
from PIL import Image


def get_links(image_type: str,
            dir_path = "/content/drive/MyDrive/Sopra Steria Next TA") -> List[str]:
    """
    Gets the different urls for the txt files

    Args:
        image_type (str): class of the image
        dir_path (str, optional): folder path. Defaults to "/content/drive/MyDrive/Sopra Steria Next TA".

    Returns:
        List[str]: list of the all the urls in the txt file
    """
    url_file_name = image_type + "_urls.txt"
    url_file_path = os.path.join(dir_path, url_file_name)
    url_file = open(url_file_path, "r")
    urls = url_file.read()
    url_list = urls.split('\n')
    return url_list

def download_images(image_type: str,
                    url_list: list,
                    dir_path: str):
    """
    Downloads the images in the urls in the url_list to an directory
    with the same name as the image class in the given directory

    Args:
        image_type (str): class of the image
        url_list (List): list of the all the image urls to be download
        dir_path (str): folder path
    """
    storage_dir_path = os.path.join(dir_path, image_type)
    # if the class folder does not exist we create it
    if not os.path.exists(storage_dir_path):
        os.mkdir(storage_dir_path)
    for i in range(len(url_list)):
        # the file name should be in this form deer_img_0.jpg containing
        # the class and index of the image
        file_name = image_type + "_img_" + str(i) + ".jpg"
        file_path = os.path.join(storage_dir_path, file_name) # We create the file path
        # We do a series of exception handling for different protocols to download
        # as much images as possible with different protocols
        # using wget, requests & urllib
        try:
            wget.download(url_list[i], out= file_path)
        except:
            try:
                res = requests.get(url_list[i], stream= True)
                if res.status_code == 200:
                    with open(file_path,'wb') as f:
                        f.write(res.content)
                else:
                    urllib.request.urlretrieve(url_list[i], file_path)
            except:
                try:
                    urllib.request.urlretrieve(url_list[i], file_path)
                except:
                    continue
                
def create_dataframe(image_types: list,
                    work_dir: str) -> pd.core.frame.DataFrame:
    """
    Creates a dataframe containing all the names of the downloaded images
    with their associated class and the image path

    Args:
        image_types (List): list of the image classes
        work_dir (str): The working folder path

    Returns:
        pd.core.frame.DataFrame: df of all the image names
    """
    image_data = pd.DataFrame()
    for image_type in image_types:
        image_dir_path = os.path.join(work_dir, image_type)
        image_names = os.listdir(image_dir_path)
        image_class = [image_type]*len(image_names)
        type_df = pd.DataFrame()
        type_df["image_name"] = image_names
        type_df["image_class"] = image_class
        type_df["image_path"] = type_df["image_name"].apply(lambda x: os.path.join(image_dir_path, x))
        image_data = pd.concat([image_data, type_df]).reset_index(drop=True)
    return image_data

def export_df(image_data: pd.core.frame.DataFrame,
            dir_path: str) -> None:
    """
    Exports the created dataframe to the designated folder

    Args:
        image_data (pd.core.frame.DataFrame): df of all the image names
        dir_path (str): folder where to store the data
    """
    # the dataframe is saved as a csv under the name all_image_data
    csv_file_path = os.path.join(dir_path, "all_image_data.csv")
    image_data.to_csv(csv_file_path, index= False)
    
def remove_corrupted_images(dir_path: str) -> None:
    """
    Removes the corrupted images from the data

    Args:
        dir_path (str): folder where to store the data
    """
    img_list = os.listdir(dir_path)
    for img_name in img_list:
        im_path = os.path.join(dir_path, img_name)
        try:
            im1= Image.open(im_path)
        except:
            os.remove(im_path)

if __name__=="__main__":
    image_types = ["boar", "deer"] # The different image classes available
    work_dir = "/content/drive/MyDrive/Sopra Steria Next TA" # The working folder path
    image_dict = dict() # We initialize a dictionary to store all the urls of each class
    for image_type in image_types:
        image_dict[image_type] = get_links(image_type)
        download_images(image_type= image_type, url_list= image_dict[image_type], dir_path= work_dir)
        remove_corrupted_images(dir_path= os.path.join(work_dir, image_type))
    image_data = create_dataframe(image_types, work_dir)
    export_df(image_data, work_dir)