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


def setup_logging():
    """

        Configures and sets up logging for the image resizer application.

        The configuration includes:
        - Creating a logger named 'image_resizer' with a logging level of INFO.
        - Creating a console handler for outputting logs to the console.
        - Creating a file handler for writing logs to a file named 'image_resize.log'.
        - Setting a log message format that includes the timestamp, logger name, log
        level, and the actual message.
        - Adding both the console handler and file handler to the logger if they haven't
        been added already.

        Returns:
            console_handler: The configured console handler.
            file_handler: The configured file handler.
            logger: The logger configured for the image resizer application.

    """
    logger = logging.getLogger("image_resizer")
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("image_resize.log")

    # Create formatters and add them to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return console_handler, file_handler, logger

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
        logging.error(
            f"Error resizing image {image_path}: {type(e).__name__} - {e}")


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


def process_images(input_dir, output_dir, size, max_workers, logger):
    """
    Processes images in a given input directory by resizing them and saving them to an
    output directory.

    Parameters:
    input_dir (str): The directory containing the images to be processed.
    output_dir (str): The directory where resized images will be saved.
    size (tuple): The desired size (width, height) for resizing the images.
    max_workers (int): The maximum number of worker threads to use for resizing images.
    logger (logging.Logger): Logger instance for logging messages during the process.

    Functionality:
    1. Finds images in the specified input directory.
    2. Logs a warning message if no images are found and exits the function.
    3. Logs the number of images found and starts the resizing process.
    4. Logs the count of images by their file extension/type.
    5. Prepares a list of tasks for resizing images, each task including the image path
    and the corresponding output path.
    6. Uses a ProcessPoolExecutor to resize images concurrently, respecting the
    max_workers limit.
    7. Times the entire resizing process and logs the time taken to resize all the
    images.
    """
    images, image_count_by_type = find_images(input_dir)

    if not images:
        logger.warning(f"No images found in the directory: {input_dir}")
        return

    logger.info(
        f"Found {len(images)} images in {input_dir}. Starting resizing process...")

    for ext, count in image_count_by_type.items():
        logger.info(f"Found {count} {ext} images.")

    image_tasks = [
        (image_path, Path(output_dir) / image_path.relative_to(input_dir))
        for image_path in images
    ]

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_image, image_path, output_path, size) for
                   image_path, output_path in image_tasks]

        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing a future: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        f"Time taken to resize {len(images)} images: {elapsed_time:.2f} seconds.")

def main():
    console_handler, file_handler, logger = setup_logging()

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

        logger.info(
            f"Starting image resizing with {args.workers} workers. Target size: "
            f"{args.size[0]}x{args.size[1]}.")

        process_images(
            args.input_dir,
            args.output_dir,
            tuple(args.size),
            args.workers,
            logger)

        logger.info("Image resizing process completed.")
    except Exception:
        logger.exception("An error occurred during the image resizing process.")
    finally:
        logger.removeHandler(console_handler)
        logger.removeHandler(file_handler)
        console_handler.close()
        file_handler.close()

if __name__ == "__main__":
    main()
