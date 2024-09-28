import os
import random
import shutil

source_dir = os.path.join("../", "data", "imagenet_224")
target_dir = r"c:\temp\imagenet_224_eval"
dirs = os.listdir(source_dir)
num_files = 100

for directory in dirs:
    files = os.listdir(os.path.join(source_dir, directory))
    selected_files = random.sample(files, min(num_files, len(files)))

    if not os.path.exists(os.path.join(target_dir, directory)):
        os.makedirs(os.path.join(target_dir, directory))

    for file in selected_files:
        source_path = os.path.join(source_dir, directory, file)
        target_path = os.path.join(target_dir, directory, file)
        shutil.move(source_path, target_path)