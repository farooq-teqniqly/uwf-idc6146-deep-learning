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
from typing import Tuple

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

def resize_image(image_path:Path, output_path:Path, size:Tuple[int,int]):
    """
    resize_image(image_path, output_path, size)

    Resizes the image located at the given image path and saves the resized image to
    the specified output path.

    Parameters:
    image_path (Path): The file path to the original image.
    output_path (Path): The file path to save the resized image.
    size (Tuple[int, int]): The target size as a tuple (width, height)

    Exceptions:
    Logs an error message if an exception occurs during the image resizing process.
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


def find_images(input_dir:Path):
    """

    find_images(input_dir)

    Searches for images in the specified directory and its subdirectories.

    Parameters:
    input_dir (Path): The directory path where the search will be performed.

    Returns:
    tuple: A tuple containing two elements:
        - list: A list of paths to the found image files.
        - defaultdict: A dictionary containing image file types as keys and their
        respective counts as values.
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_count_by_type = defaultdict(int)
    image_paths = []

    for file in input_dir.rglob("*"):
        if file.suffix.lower() in image_extensions:
            image_paths.append(file)
            image_count_by_type[file.suffix.lower()] += 1  # Count each image type

    return image_paths, image_count_by_type


def process_images(
        input_dir:Path,
        output_dir:Path,
        size:Tuple[int,int],
        logger,
        max_workers:int=1):
    """
    Processes images by resizing them to the specified size and saving them to the
    output directory.

    Args:
        input_dir (Path): Directory containing input images.
        output_dir (Path): Directory where resized images will be saved.
        size (Tuple[int, int]): Target size for resizing images.
        logger: Logger instance used for logging information and errors.
        max_workers (int, optional): Maximum number of worker threads for parallel
        processing. Defaults to 1.

    Returns:
        None
    """

    if not all(dim > 0 for dim in size):
        msg = "Values of size tuple must be greater than 0."
        raise ValueError(msg)

    if max_workers < 1:
        msg = "Value must be greater than 0 for max_workers."
        raise ValueError(msg)

    images, image_count_by_type = find_images(input_dir)

    if not images:
        logger.warning(f"No images found in the directory: {input_dir}")
        return

    logger.info(
        f"Found {len(images)} images in {input_dir}. Starting resizing process...")

    for ext, count in image_count_by_type.items():
        logger.info(f"Found {count} {ext} images.")

    image_tasks = [
        (image_path, output_dir / image_path.relative_to(input_dir))
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
