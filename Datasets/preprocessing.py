import torchvision
from torchvision import transforms
from torchvision.transforms import v2
import os
from PIL import Image

def prepare_transforms():
    """
    prepares the data transforms and data augmentations

    Returns:
        Tuple: train & test transforms
    """
    # Create training transforms with data augmentation
    train_transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.RandomHorizontalFlip(p = 0.2),  # Data Augmentation
        v2.RandomVerticalFlip(p = 0.2),    # Data Augmentation
        v2.RandomGrayscale(p = 0.2),       # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Create testing transform (no data augmentation)
    test_transform = v2.Compose([ # the test transforms follows the train transforms
                                    # except the data augmentation
        v2.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

def remove_transparency(dir_path: str) -> None:
    """
    remove the transparent part of the images
    since models do not fare well with the RGBA
    type images

    Args:
        dir_path (str): folder path
    """
    paths = os.listdir(dir_path) # get the images in the folder
    for path in paths:
        img_path = os.path.join(dir_path, path) # get the image path
        png = Image.open(img_path) # Open the image
        # Check if the image has a transparent part
        if png.mode in ('RGBA', 'LA') or (png.mode == 'P' and 'transparency' in png.info):
            background = Image.new('RGBA', png.size, (255,255,255)) # Create a white background with the RGBA mode
            try:
                alpha_composite = Image.alpha_composite(background, png) # Overlay the original image with the bg
            except:
                # Since some images have transparent parts but aren't of RGBA mode
                # we need to convert them to RGBA
                png = png.convert("RGBA")
                alpha_composite = Image.alpha_composite(background, png)
            alpha_composite = alpha_composite.convert("RGB")
            alpha_composite.save(img_path, 'JPEG', quality=80) # We export the resulting image
            
def fix_all_transparent_images(train_dir_path: str,
                                test_dir_path: str,
                                image_types: list) -> None:
    """
    We go through all the image folders and we fix the
    transparent images

    Args:
        image_types (List[str]): list of the image classes
        train_dir_path (str): path to the train directory
        test_dir_path (str): path to the test directory
    """
    for img_class in image_types:
        class_train_dir = os.path.join(train_dir_path, img_class)
        class_test_dir = os.path.join(test_dir_path, img_class)
        remove_transparency(class_train_dir)
        remove_transparency(class_test_dir)
