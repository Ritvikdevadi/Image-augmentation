import albumentations as A
import cv2
import os
transform = A.Compose([
        A.Blur(blur_limit=(3, 7)),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.8),
        A.ImageCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
        A.InvertImg(p=0.6),
        ])




folder_path = 'Document dataset\\Id-Dataset\\aadhar'  
image_paths = []

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image_paths.append(image_path)


augmented_images = []

for path in image_paths:
    image = cv2.imread(path)
    augmented = transform(image=image)['image']
    augmented_images.append(augmented)

for augmented_image in augmented_images:
    cv2.imshow("Augmented Image", augmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

