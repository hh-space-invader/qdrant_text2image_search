# for system calls (file/folder operations)
import os, shutil
# to download dataset
import requests
# to extract zip file
import zipfile
# to load images from disk
from PIL import Image
# progress bar
from tqdm import tqdm
# for basic ops
import numpy as np
import pandas as pd

def download_extract_dataset():

    # Zalando Viton-HD dataset
    data_url = "https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip"

    # if the zip dataset not exist on disk
    if not os.path.exists("../data/raw/zalando-hd-resized.zip"):
        # go to dropbox
        response = requests.get(
            data_url,
            stream=True
        )

        # save zip to disk
        with open("../data/raw/zalando-hd-resized.zip", 'wb') as buff:
            for chunk in response.iter_content(128):
                buff.write(chunk)

    # extract data
    with zipfile.ZipFile(file="../data/raw/zalando-hd-resized.zip", mode="r") as buff:
        buff.extractall(path="../data/raw/zalando-hd-resized")

    # copy contents of cloth folder to raw
    shutil.copytree(
        src="../data/raw/zalando-hd-resized/test/cloth/",
        dst="../data/raw/",
        dirs_exist_ok=True)

    # delete unneccesary folder/zip 
    shutil.rmtree(path="../data/raw/zalando-hd-resized")
    os.remove(path="../data/raw/zalando-hd-resized.zip")

def embed_dataset(
        num_of_samples:int,
        images_paths:str,
        data_processor,
        embedder
) -> pd.DataFrame:
    images_embeddings = list()
    
    # loop over all images in folder
    for idx, img in tqdm(
        iterable=enumerate(images_paths[:num_of_samples]),
        desc="embedding all images",
        total=len(images_paths[:num_of_samples]), 

    ):
        # load image from disk
        image = Image.open(
            fp=img,
            mode="r"
        )

        # process image image
        processed_image = data_processor(
        images=image,
        text=None,
        return_tensors="tf"
        )["pixel_values"]

        # embed image (latent vectors)
        image_embedding = np.squeeze(embedder.get_image_features(processed_image).numpy()) # convert from eager tensor to numpy

        # append to list
        images_embeddings.append({"id":idx, "embedding":image_embedding, "dir":{"dir":img}})
    
    # convert result as dataframe and return
    return  pd.DataFrame.from_dict(images_embeddings)