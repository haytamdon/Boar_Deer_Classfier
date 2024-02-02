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
                                        create_dataloaders)
from utils.utils import (get_miscellanous_variables,
                        get_device,
                        model_summary,
                        visualize_training_accuracy,
                        visualize_training_loss)
from Models.efficient_net_model import (get_efficient_net_model_weights,
                                        get_model_summary,
                                        get_model_transforms,
                                        update_model,
                                        define_model)
from Models.vgg_model import TinyVGG
from train_val_loops.train_val_functions import (train,
                                                train_advanced)
import os
import torch
from torch import nn

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
    
    # Choose Device
    device = get_device()
    
    # Defining first model
    vgg_model = TinyVGG(input_shape=3, # number of color channels (3 for RGB)
                        hidden_units=10,
                        output_shape=len(train_data.classes))
    
    # Getting the model summary
    print(model_summary(vgg_model, [1, 3, 256, 256]))
    
    # Loading the model into the device
    model = vgg_model.to(device)
    # Defining the optimiser
    optim = torch.optim.SGD(params= model.parameters(), lr=0.001)
    # Running the training loop
    metrics = train(model = model, train_dataloader= train_dataloader, test_dataloader= test_dataloader, optimizer= optim,
        device= device)
    
    # Plotting the results
    visualize_training_accuracy(metrics, 5) # Plotting Accuracy
    visualize_training_loss(metrics, 5) # Plotting loss
    
    # New model Efficient Net
    eff_net_weights = get_efficient_net_model_weights() # We get the model weights
    efficient_net_transforms = get_model_transforms(eff_net_weights) # We get the model transforms
    # Recreating the datasets with the updated transforms
    train_data, test_data = create_datasets(train_dir= train_dir_path,
                    test_dir= test_dir_path,
                    train_transform= efficient_net_transforms,
                    # We use the same transforms of the trainset
                    # in the testset since there are no data augmentations
                    test_transform= efficient_net_transforms)
    # We recreated the dataloaders
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data)
    class_names, class_dict = get_miscellanous_variables(train_data)
    
    # efficient model definition
    model = define_model()
    model = model.to(device)
    
    model = update_model(model, class_names, device)
    # Summary of model after updating the classifier the block
    get_model_summary(model)
    
    # Defining Loss function
    loss_fn = nn.CrossEntropyLoss()
    # Defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # definining number of epochs
    num_epochs = 10
    # Training the model
    results = train_advanced(model= model,
        train_dataloader= train_dataloader,
        test_dataloader= test_dataloader,
        optimizer= optimizer,
        device= device,
        epochs= num_epochs)
    
    # Visualize results
    visualize_training_accuracy(results, num_epochs) # Plotting Accuracy
    visualize_training_loss(results, num_epochs) # Plotting loss
    