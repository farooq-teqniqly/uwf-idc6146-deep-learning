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

logger = logging.getLogger("image_resizer")
logger.setLevel(logging.INFO)

def setup_logging():
    """
    Configures and sets up logging for the image resizer.

    This function creates a FileHandler for logging to a file named
    'image_resizer.log'. It then creates a Formatter to format the log
    messages with the time, logger name, log level, and message. This
    formatter is set for the handler, which is then added to the logger.

    Returns:
        logging.FileHandler: The file handler configured for logging.
    """
    handler = logging.FileHandler("image_resizer.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return handler

def resize_image(image_path, output_path, size):
    """
    Resize an image to a specified size and save it to a given path.

    Parameters:
    image_path (str or Path): Path to the input image file
    output_path (str or Path): Path to save the resized image
    size (tuple): Desired size for the resized image as (width, height)

    Logs output:
    Logs informational messages about the status and errors encountered during resizing.

    Exceptions:
    Catches general exceptions and logs an error message in case of failures.
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

    find_images(input_dir)

    Searches for images in the specified directory and its subdirectories.

    Parameters:
    input_dir (str): The directory path where the search will be performed.

    Returns:
    tuple: A tuple containing two elements:
        - list: A list of paths to the found image files.
        - defaultdict: A dictionary containing image file types as keys and their
        respective counts as values.
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
    Processes and resizes images from the input directory
    and saves them to the output directory.

    Arguments:
    input_dir (str): The directory to search for images to process.
    output_dir (str): The directory where the processed images will be saved.
    size (tuple): The target size for the resized images, specified as
    a (width, height) tuple.
    max_workers (int): The maximum number of worker processes
    to use for parallel processing.

    Logs:
    Warnings if no images are found in the input directory.
    Information about the number of images found and their types.
    Timing information for the resizing process.
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
    handler = setup_logging()
    try:
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
    finally:
        logger.removeHandler(handler)
        handler.close()

if __name__ == "__main__":
    main()
