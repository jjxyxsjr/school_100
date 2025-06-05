import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_yolo_dataset(image_dir, label_dir, output_root='ccpd_yolo', val_ratio=0.2):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    train_files, val_files = train_test_split(image_files, test_size=val_ratio, random_state=42)

    for split, files in [('train', train_files), ('val', val_files)]:
        img_out = os.path.join(output_root, 'images', split)
        lbl_out = os.path.join(output_root, 'labels', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for f in files:
            shutil.copy(os.path.join(image_dir, f), os.path.join(img_out, f))
            label_file = f.replace('.jpg', '.txt')
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(lbl_out, label_file))

prepare_yolo_dataset('image', 'labels')
