"""
Module: download_synsets

This module provides functionality to download and rename synset tar files from
ImageNet.
It utilizes asynchronous operations to handle multiple downloads concurrently, making
the download process efficient and quicker.

Logging is configured to display info level messages which will help in tracking the
progress of downloads.

The module defines a main function which orchestrates the download process for all the
synsets specified in the synset_mapping dictionary. Each synset is downloaded and
renamed according to its CIFAR-10 class name.

The primary components of this module include:

- ROOT_URL: The base URL from where the ImageNet tar files are downloaded.
- synset_mapping: A dictionary mapping CIFAR-10 class names to their corresponding
  synset identifiers in ImageNet.
- download_dir: The directory where the downloaded tar files will be stored.

Functions:
- download_and_rename_tar: This function handles the actual downloading and renaming
  of a synset tar file using aiohttp for asynchronous HTTP requests.
- main: The main function that initializes the aiohttp ClientSession and triggers
  the download of all synsets concurrently.

Usage:
Run the module as a standalone script to initiate the download process:
    $ python download_synsets.py

Dependencies:
- aiohttp: Asynchronous HTTP client/server framework
- asyncio: Python standard library module for writing asynchronous code
- logging: Python standard library module for logging messages
- os: Python standard library module for operating system-related functionalities
"""
import argparse
import asyncio
import logging
import os

import aiohttp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

ROOT_URL = "https://image-net.org/data/winter21_whole/"

synset_mapping = {
    "airplane": "n02691156",
    "automobile": "n02958343",
    "bird": "n01503061",
    "cat": "n02121808",
    "deer": "n02419796",
    "dog": "n02084071",
    "frog": "n01641577",
    "horse": "n02374451",
    "ship": "n04194289",
    "truck": "n04467665",
}

async def download_and_rename_tar(session, synset_id, cifar_class_name, download_dir):
    """
    Asynchronously downloads a .tar file from a given URL, renames the file,
    and saves it to a specified directory.

    Parameters:
    session (aiohttp.ClientSession): The aiohttp session used for making HTTP requests.
    synset_id (str): The synset ID used to construct the download URL.
    cifar_class_name (str): The name associated with the CIFAR class, used to rename
    the downloaded file.
    download_dir (str): The directory where the downloaded file will be saved.

    Behavior:
    - Constructs the download URL using the provided synset_id.
    - Constructs the renamed filename using synset_id and cifar_class_name,
    then checks if the file already exists in download_dir.
    - If the renamed file exists, logs a message and returns without downloading.
    - If the renamed file does not exist, makes an async HTTP GET request
    to download the tar file.
    - Saves the downloaded file to download_dir with a temporary filename.
    - Renames the file to the constructed renamed filename.
    - Logs progress and any errors encountered during the download and renaming process.
    """
    tar_url = f"{ROOT_URL}{synset_id}.tar"
    renamed_tar_filename = f"{synset_id}-{cifar_class_name}.tar"
    renamed_tar_file_path = os.path.join(download_dir, renamed_tar_filename)

    if os.path.exists(renamed_tar_file_path):
        logging.info(f"{renamed_tar_filename} already exists, skipping download.")
        return

    try:
        async with session.get(tar_url) as response:
            if response.status == 200:
                logging.info(f"Downloading {tar_url}...")
                tar_file_path = os.path.join(download_dir, f"{synset_id}.tar")
                with open(tar_file_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)

                logging.info(f"Downloaded {tar_file_path}")
                os.rename(tar_file_path, renamed_tar_file_path)
                logging.info(f"Renamed {tar_file_path} to {renamed_tar_filename}")
            else:
                logging.error(
                    f"Failed to download {tar_url}, status code: {response.status}")
    except Exception as e:
        logging.error(f"An error occurred while downloading {tar_url}: {e}")


async def main(download_dir: str):
    """
        Main function to download and rename CIFAR tar files.

        Args:
            download_dir (str): Directory to download the tar files.

        This function checks if the specified download directory exists; if not,
        it creates the directory.
        It then initializes an asynchronous HTTP session and creates a list of
        download tasks
        for each CIFAR class and synset ID as specified in the synset_mapping
        dictionary.
        Finally, it concurrently executes these download tasks using asyncio.gather.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for cifar_class, synset_id in synset_mapping.items():
            tasks.append(download_and_rename_tar(
                session, synset_id, cifar_class, download_dir))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ImageNet synsets and rename them with CIFAR-10 "
                    "class names.")

    parser.add_argument("--download-dir", type=str, required=True,
                        help="Directory where tar files will be downloaded.")

    args = parser.parse_args()
    asyncio.run(main(args.download_dir))