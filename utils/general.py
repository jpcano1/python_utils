import numpy as np
import requests
import sys
from tqdm.auto import tqdm
import zipfile
import os
import matplotlib.pyplot as plt
import zipfile
import tarfile
import pickle

"""
OS Functions
"""
def create_and_verify(*args):
    full_path = os.path.join(*args)
    exists = os.path.exists(full_path)
    if exists:
        return full_path
    else:
        raise FileNotFoundError("La ruta no existe")

def read_listdir(dir_):
    listdir = os.listdir(dir_)
    full_dirs = list()
    for d in listdir:
        full_dir = create_and_verify(dir_, d)
        full_dirs.append(full_dir)
    return np.sort(full_dirs)

def extract_file(filename: str, dst=None):
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename) as zfile:
            print("\nExtracting Zip File...")
            zfile.extractall(dst)
            zfile.close()
    elif ".tar" in filename:
        with tarfile.open(filename, "r") as tfile:
            print("\nExtracting Tar File...")
            tfile.extractall(dst)
            tfile.close()
    print("Deleting File...")
    os.remove(filename)

def unpickle(filename):
    with open(filename, "rb") as fo:
        pickle_data = pickle.load(fo, encoding='bytes')
        fo.close()
    return pickle_data

"""
DataViz Functions
"""
def imshow(img, title=None, color=True, cmap="gray", 
            axis=False, ax=None):
    if not ax:
        ax = plt
    # Plot Image
    if color:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap)

    # Ask about the axis
    if not axis:
        ax.axis("off")

    # Ask about the title
    if title:
        ax.title(title)

def visualize_subplot(imgs: list, titles: list, 
                    division: tuple, figsize: tuple=None, cmap="gray"):
    """
    An even more complex function to plot multiple images in one or
    two axis
    :param imgs: The images to be shown
    :param titles: The titles of each image
    :param division: The division of the plot
    :param cmap: Image Color Map
    :param figsize: the figsize of the entire plot
    """
    # We create the figure
    fig: plt.Figure = plt.figure(figsize=figsize)

    # Validate the figsize
    if figsize:
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

    # We make some assertions, the number of images and the number of titles
    # must be the same
    assert len(imgs) == len(titles), "La lista de imágenes y de títulos debe ser del mismo tamaño"

    # The division must have sense w.r.t. the number of images
    assert np.prod(division) >= len(imgs)

    # A loop to plot the images
    for index, title in enumerate(titles):
        ax: plt.Axes = fig.add_subplot(division[0], 
                            division[1], index+1)
        ax.imshow(imgs[index], cmap=cmap)
        ax.set_title(title)
        plt.axis("off")

"""
Miscellaneous Functions
"""
def download_content(url, filename, dst="./data", chnksz=1000):
    try:
        r = requests.get(url, stream=True)
    except Exception as e:
        print("Error de conexión con el servidor")
        sys.exit()
        
    full_path = os.path.join(dst, filename)
    if not os.path.exists(dst):
        os.makedirs(dst)

    with open(full_path, "wb") as f:
        try:
            total = int(np.ceil(int(r.headers.get("content-length"))/chnksz))
        except:
            total = 0

        gen = r.iter_content(chunk_size=chnksz)

        for pkg in tqdm(gen, total=total, unit="KB"):
            f.write(pkg)
        f.close()
        r.close()
    
    extract_file(full_path, dst)

def download_file_from_google_drive(id_, filename, dst="./data", size=None,
                                    chnksz=1000):
    """
    Retrieved and Improved from https://stackoverflow.com/a/39225039
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, filename, dst, size=None,
                              chnksz=1000):
        full_path = os.path.join(dst, filename)
        if not os.path.exists(dst):
            os.makedirs(dst)
        with open(full_path, "wb") as f:
            gen = response.iter_content(chunk_size=chnksz)
            for chunk in tqdm(gen, total=size, unit="KB"):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
            f.close()

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={ 'id' : id_ }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id_, 'confirm' : token }
        response = session.get(url, params=params, stream=True)

    save_response_content(response, filename, dst, size=size,
                          chnksz=chnksz)
    response.close()
    full_path = os.path.join(dst, filename)

    extract_file(full_path, dst)
    return

"""
Data Scalers
"""
def scale(img, min, max, dtype="uint8"):
    img_min = img.min()
    img_max = img.max()
    m = (max - min) / (img_max - img_min)
    return (m * (img - img_min) + min).astype(dtype)