from PIL import Image
import numpy as np
import os
import glob
import random
from imgaug import augmenters as iaa


def augment_class(class_name):
    os.chdir(class_name)

    all_images = glob.glob('*.jpg')
    augmented_images = glob.glob('*_.jpg')  # already existing, augmented images on the hardware
    all_img_amt = len(all_images)
    augmt_img_amt = len(augmented_images)
    req_augmt = 3000 - all_img_amt  # number of missing datapoints, compared to desired amount

    img_batch = []
    print(f"Reading images of class {class_name}...")
    for i in range(req_augmt):
        chosen_img = all_images[random.randint(0, all_img_amt-augmt_img_amt-1)]
        img = np.array(Image.open(os.path.join(os.path.abspath('.'), chosen_img)))
        img_batch.append(img)
        if (i+1) % 10 == 0:  # augment & save images in batches of 10 to consume less memory
            # print("Augmentation in process...")
            img_batch = aug.augment_images(img_batch)   # augment_images() accepts a 4D np.array or list of 3D np.arrays
            # print(f"Saving images...")
            for idx, new_img in enumerate(img_batch):
                Image.fromarray(new_img).save(f"{all_img_amt+i+idx+1}_.jpg")
            print(f"{class_name} class batch augmentation complete.\nBatch count: {i//10}\n")
            img_batch = []

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
        iaa.imgcorruptlike.ImpulseNoise(severity=random.randint(1, 3)),
        iaa.imgcorruptlike.ElasticTransform(severity=random.randint(1, 3)),
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
