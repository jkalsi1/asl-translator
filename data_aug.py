import os
import cv2
import numpy as np
from pathlib import Path

def augment_brightness(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    aug = np.random.uniform(0.25, 2)
    img[:, :, 2] = img[:, :, 2] * aug
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def augment_contrast(image):
    contrast = np.random.uniform(0.25, 2)
    return cv2.convertScaleAbs(image, alpha=contrast)

def augment_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 0.1 * np.random.uniform(0, 1)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss.reshape(row, col, ch)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def augment_scale(image):
    scale = np.random.uniform(0.25, 2)
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    return cv2.resize(image, (new_width, new_height))

def augment_shear(image):
    rows, cols, ch = image.shape
    shear_factor = np.random.uniform(-0.5, 0.5)
    M = np.array([[1, shear_factor, 0],
                    [0, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

def save_img(image, output_path, suffix):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{Path(output_path).stem}_{suffix}{Path(output_path).suffix}"
    cv2.imwrite(str(output_dir / output_filename), image)

def main():
    og_imgs = 'asl_dataset'
    out = 'asl_dataset_augmented'
    Path(out).mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(og_imgs):
        for file in files:
            if file.endswith(('.jpeg')):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                path = os.path.relpath(file_path, og_imgs)
                save_img(image, os.path.join(out, path), 'original')

                for i in range(5):
                    brightness_image = augment_brightness(image)
                    contrast_image = augment_contrast(image)
                    noise_image = augment_noise(image)
                    scale_image = augment_scale(image)
                    sheer_image = augment_shear(image)

                    save_img(brightness_image, os.path.join(out, path), f'brightness_{i}')
                    save_img(contrast_image, os.path.join(out, path), f'contrast_{i}')
                    save_img(noise_image, os.path.join(out, path), f'noise_{i}')
                    save_img(scale_image, os.path.join(out, path), f'scale_{i}')
                    save_img(sheer_image, os.path.join(out, path), f'sheer_{i}')
        
    
if __name__ == "__main__":
    main()