import os
import argparse
import logging
import time
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_resizer.log"),
        logging.StreamHandler()
    ]
)


# Function to resize an image
def resize_image(image_path, output_path, size):
    try:
        with Image.open(image_path) as img:
            resized_img = img.resize(size,  Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            resized_img.save(output_path)
        logging.info(f"Resized and saved: {output_path}")
    except Exception as e:
        logging.error(f"Error resizing image {image_path}: {e}")


def find_images(input_dir):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_count_by_type = defaultdict(int)
    image_paths = []

    for file in Path(input_dir).rglob('*'):
        if file.suffix.lower() in image_extensions:
            image_paths.append(file)
            image_count_by_type[file.suffix.lower()] += 1  # Count each image type

    return image_paths, image_count_by_type


def process_images(input_dir, output_dir, size, max_workers):
    images, image_count_by_type = find_images(input_dir)

    if not images:
        logging.warning(f"No images found in the directory: {input_dir}")
        return

    logging.info(
        f"Found {len(images)} images in {input_dir}. Starting resizing process...")

    for ext, count in image_count_by_type.items():
        logging.info(f"Found {count} {ext} images.")

    image_tasks = [
        (image_path, Path(output_dir) / image_path.relative_to(input_dir))
        for image_path in images
    ]

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_image, image_path, output_path, size) for
                   image_path, output_path in image_tasks]

        for future in futures:
            future.result()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(
        f"Time taken to resize {len(images)} images: {elapsed_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(
        description='Resize images in a folder recursively.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the input folder containing images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output folder where resized images will be saved')
    parser.add_argument('--size', type=int, nargs=2, default=(224, 224),
                        help='Target size for the resized images (width height)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers to use for resizing')

    args = parser.parse_args()

    logging.info(
        f"Starting image resizing with {args.workers} workers. Target size: {args.size[0]}x{args.size[1]}.")
    process_images(args.input_dir, args.output_dir, tuple(args.size), args.workers)
    logging.info("Image resizing process completed.")


if __name__ == '__main__':
    main()
