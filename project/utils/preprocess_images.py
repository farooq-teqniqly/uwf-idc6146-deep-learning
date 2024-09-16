"""
Module for preprocessing images.

This module includes functions to resize and find images, as well as a main function
to process images as a whole.

Functions:
    - resize_image: Resizes an image to given dimensions.
    - find_images: Identifies and retrieves images from a specified directory.
    - process_images: Handles the overall processing workflow for images.
    - main: Entry point for the module, coordinating the complete image
    preprocessing pipeline.
"""
import argparse
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image_resizer.log"),
        logging.StreamHandler(),
    ],
)


def resize_image(image_path, output_path, size):
    """
    Resize an image.
    @param image_path: The file path of the input image to be resized.
    @param output_path: The file path where the resized image will be saved.
    @param size: The dimensions to resize the image to, specified as
    a tuple (width, height).
    @return: None
    """
    try:
        with Image.open(image_path) as img:
            resized_img = img.resize(size,  Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            resized_img.save(output_path)
        logging.info(f"Resized and saved: {output_path}")
    except Exception as e:
        logging.error(f"Error resizing image {image_path}: {e}")


def find_images(input_dir):
    """
    Recursively find images to resize.
    @param input_dir: Directory to search for image files.
    @return: A tuple containing a list of image paths and a dictionary
    with the count of each image type.
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_count_by_type = defaultdict(int)
    image_paths = []

    for file in Path(input_dir).rglob("*"):
        if file.suffix.lower() in image_extensions:
            image_paths.append(file)
            image_count_by_type[file.suffix.lower()] += 1  # Count each image type

    return image_paths, image_count_by_type


def process_images(input_dir, output_dir, size, max_workers):
    """
    This method does the work of resizing images.
    @param input_dir: Directory path where input images are stored
    @param output_dir: Directory path where resized images will be stored
    @param size: Desired size (width, height) of the resized images
    @param max_workers: Maximum number of worker processes to use for resizing
    @return: None
    """
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
        description="Resize images in a folder recursively.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the input folder containing images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the output folder where resized images "
                             "will be saved")
    parser.add_argument("--size", type=int, nargs=2, default=(224, 224),
                        help="Target size for the resized images (width height)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers to use for resizing")

    args = parser.parse_args()

    logging.info(
        f"Starting image resizing with {args.workers} workers. Target size: "
        f"{args.size[0]}x{args.size[1]}.")
    process_images(args.input_dir, args.output_dir, tuple(args.size), args.workers)
    logging.info("Image resizing process completed.")


if __name__ == "__main__":
    main()
