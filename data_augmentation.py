from PIL import Image
import numpy as np
import os
import glob
import random
from imgaug import augmenters as iaa


def augment_class(class_name):
    os.chdir(class_name)

    img_names = glob.glob('*.jpg')
    img_amt = len(img_names)
    req_augmt = 1500 - img_amt  # number of missing datapoints, compared to desired amount

    img_batch = []
    print(f"Reading images of class {class_name}...")
    for i in range(req_augmt):
        chosen_img = img_names[random.randint(0, len(img_names)-1)]
        img = np.array(Image.open(os.path.join(os.path.abspath('.'), chosen_img)))
        img_batch.append(img)
    print("Augmentation in process...")
    img_batch = aug.augment_images(img_batch)
    print("Saving images...")
    for idx, new_img in enumerate(img_batch):
        Image.fromarray(new_img).save(f"{img_amt+idx}_.jpg")
    print(f"{class_name} class augmentation complete.\n")

    os.chdir("..")


aug = iaa.SomeOf(3, [
    iaa.OneOf([
        iaa.Affine(scale=(0.5, 1.5)),
        iaa.ScaleX((0.5, 1.5)),
        iaa.ScaleY((0.5, 1.5)),
    ]),
    iaa.OneOf([
        iaa.Rotate((-180, 180)),
        iaa.ShearX((-60, 60)),
        iaa.ShearY((-60, 60)),
    ]),
    iaa.OneOf([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ]),
    iaa.OneOf([
        iaa.AverageBlur(k=(2, 6)),
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.Sharpen(alpha=(0.0, 0.8), lightness=(0.8, 1.2)),
    ]),
    iaa.OneOf([
        iaa.imgcorruptlike.GaussianNoise(severity=random.randint(1, 3)),
        iaa.imgcorruptlike.ImpulseNoise(severity=random.randint(1, 3))
    ]),
    iaa.OneOf([
        iaa.AveragePooling(2),
        iaa.MaxPooling(2),
        iaa.MinPooling(2),
    ]),
    iaa.OneOf([
        iaa.pillike.Equalize(),
        iaa.pillike.Autocontrast(),
        iaa.pillike.FilterSmooth(),
    ]),
])

# paths to operate with during augmentation 
start_path = os.path.abspath(f'.{os.path.sep}Dataset{os.path.sep}train_augmented')
os.chdir(start_path)

# folder names (classes)
classes = ["Bread", "Dairy_product", "Dessert", "Egg",
           "Fried_food", "Meat", "Noodles_Pasta", "Seafood",
           "Soup", "Vegetable_Fruit"]

augment_class(classes[0])
augment_class(classes[1])
augment_class(classes[2])
augment_class(classes[3])
augment_class(classes[4])
augment_class(classes[5])
augment_class(classes[6])
augment_class(classes[7])
augment_class(classes[8])
augment_class(classes[9])
