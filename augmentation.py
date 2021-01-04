from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
import cv2
from albu import IsotropicResize
size = 256
# Declare an augmentation pipeline

transform = Compose([
    ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    GaussNoise(p=0.1),
    GaussianBlur(blur_limit=3, p=0.05),
    HorizontalFlip(p=0.5),
    OneOf([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
    ], p=0.7),
    PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
    ToGray(p=0.2),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
])


# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("/home/ubuntu/dataset/dfdc_image/train/dfdc_train_part_0/aaqaifqrwn/frame0.jpg")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]

cv2.imwrite('image.jpg', transformed_image)