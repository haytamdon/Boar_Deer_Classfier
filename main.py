from Download_data.data_download import (get_links, 
                                        download_images, 
                                        remove_corrupted_images, 
                                        create_dataframe, 
                                        export_df)
from Datasets.organising_data import (copy_images_to_appropriate_folder, 
                                    create_folders, 
                                    split_train_test_images)
from Datasets.preprocessing import (fix_all_transparent_images,
                                    prepare_transforms)
from Datasets.create_datasets import (create_datasets,
                                        create_dataloaders,
                                        get_miscellanous_variables)
import os
import torch

if __name__=="__main__":
    # Creating Necessary Variables
    image_types = ["boar", "deer"] # The different image classes available
    work_dir = "/content/drive/MyDrive/Sopra Steria Next TA" # The working folder path
    train_dir_path = "/content/drive/MyDrive/Sopra Steria Next TA/train" # path of the training data folder
    test_dir_path = "/content/drive/MyDrive/Sopra Steria Next TA/test" #path of the test data folder
    image_dict = dict() # We initialize a dictionary to store all the urls of each class
    
    # Set Seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Data Download
    for image_type in image_types:
        image_dict[image_type] = get_links(image_type)
        download_images(image_type= image_type, url_list= image_dict[image_type], dir_path= work_dir)
        remove_corrupted_images(dir_path= os.path.join(work_dir, image_type))
    image_data = create_dataframe(image_types, work_dir)
    export_df(image_data, work_dir)
    
    # Organising the downloaded data
    create_folders(image_types, train_dir_path, test_dir_path)
    df_train, df_test = split_train_test_images(image_data)
    copy_images_to_appropriate_folder(df_train, df_test, train_dir_path, test_dir_path, image_types)
    
    # Preprocess data
    fix_all_transparent_images(train_dir_path, test_dir_path, image_types)
    train_transform, test_transform = prepare_transforms()
    
    # Create datasets
    train_data, test_data = create_datasets(train_dir_path, test_dir_path, train_transform, test_transform)
    class_names, class_dict = get_miscellanous_variables(train_data)
    
    # Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data, 32)
    