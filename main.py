from Download_data.data_download import get_links, download_images, remove_corrupted_images, create_dataframe, export_df
import os

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