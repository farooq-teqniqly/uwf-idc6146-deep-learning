import os
import aiohttp
import asyncio
import logging

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
    "truck": "n04467665"
}

download_dir = r"C:\Users\faroo\Downloads\imagenet_cifar10_tars"

if not os.path.exists(download_dir):
    os.makedirs(download_dir)


async def download_and_rename_tar(session, synset_id, cifar_class_name):
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
                with open(tar_file_path, 'wb') as f:
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


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for cifar_class, synset_id in synset_mapping.items():
            tasks.append(download_and_rename_tar(session, synset_id, cifar_class))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
