import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import requests
import sys
import tarfile
from tqdm.auto import tqdm
from typing import List, Tuple
import zipfile

"""
OS Functions
"""


def read_listdir(dir_):
    """
    Function that returns the fullpath of each dir in
    the parameter
    :param dir_: the non-empty directory
    :return: the list of fulldirs in the directory
    """
    listdir = os.listdir(dir_)
    full_dirs = list()
    for d in listdir:
        # Concatenate each dir
        if isinstance(d, bytes):
            d = d.decode("utf-8")
        full_dir = os.path.join(dir_, d)
        full_dirs.append(full_dir)
    return np.sort(full_dirs)


def create_and_verify(*args: List[str], list_=False):
    """
    Function that creates a directory and verifies
    its existence
    :param args: the parts of the path
    :param list_: boolean that determines if the user
    wants to return a list
    :return: The path checked
    """
    full_path = os.path.join(*args)
    exists = os.path.exists(full_path)
    if exists:
        if list_:
            return read_listdir(full_path)
        return full_path
    else:
        raise FileNotFoundError("La ruta no existe")


def extract_file(filename: str, dst: str = None):
    flag = False
    if filename.endswith(".zip"):
        flag = True
        with zipfile.ZipFile(filename) as zfile:
            bar = tqdm(zfile.namelist())
            bar.set_description("Extracting File")
            for file_ in bar:
                zfile.extract(file_, dst)
    elif ".tar" in filename:
        flag = True
        with tarfile.open(filename, "r") as tfile:
            bar = tqdm(tfile.getnames())
            bar.set_description("Extracting File")
            for file_ in bar:
                tfile.extract(file_, dst)
    if flag:
        print("Deleting File...")
        os.remove(filename)


def unpickle(filename: str):
    with open(filename, "rb") as fo:
        pickle_data = pickle.load(fo, encoding='bytes')
    return pickle_data


"""
DataViz Functions
"""


def imshow(
    img: np.ndarray,
    title: str = None,
    color: bool = True,
    cmap: str = "gray",
    axis: bool = False,
    ax: plt.Axes = None,
):
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


def visualize_subplot(
    imgs: List[np.ndarray],
    titles: List[str],
    division: Tuple[int],
    figsize: Tuple[int] = None,
    cmap: str = "gray",
):
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
        ax: plt.Axes = fig.add_subplot(division[0], division[1], index + 1)
        ax.imshow(imgs[index], cmap=cmap)
        ax.set_title(title)
        plt.axis("off")


"""
Miscellaneous Functions
"""


def download_content(url: str, filename: str, dst: str = "./data", chnksz: int = 1000):
    try:
        r = requests.get(url, stream=True)
    except Exception as _:
        print("Error de conexión con el servidor")
        sys.exit()

    full_path = os.path.join(dst, filename)
    if not os.path.exists(dst):
        os.makedirs(dst)

    with open(full_path, "wb") as f:
        try:
            total = int(np.ceil(int(r.headers.get("content-length")) / chnksz))
        except:
            total = 0

        gen = r.iter_content(chunk_size=chnksz)

        for pkg in tqdm(gen, total=total, unit="KB"):
            f.write(pkg)
        r.close()

    extract_file(full_path, dst)


def download_file_from_google_drive(
    id_: str, filename: str, dst: str = "./data", size: float = 0, chnksz: int = 1000
):
    """
    Retrieved and Improved from https://stackoverflow.com/a/39225039
    """

    def get_confirm_token(response: requests.Response) -> str:
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(
        response: requests.Response, filename: str, dst: str, size: float = 0, chnksz: int = 1000
    ):
        full_path = os.path.join(dst, filename)
        if not os.path.exists(dst):
            os.makedirs(dst)
        with open(full_path, "wb") as f:
            gen = response.iter_content(chunk_size=chnksz)
            for chunk in tqdm(gen, total=size, unit="KB"):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={'id': id_}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id_, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, filename, dst, size=size, chnksz=chnksz)
    response.close()
    full_path = os.path.join(dst, filename)

    extract_file(full_path, dst)


"""
Data Scalers
"""


def scale(img: np.ndarray, min_: float, max_: float, dtype: str = "uint8") -> np.ndarray:
    img_min = img.min()
    img_max = img.max()
    m = (max_ - min_) / (img_max - img_min)
    return (m * (img - img_min) + min_).astype(dtype)


def std_scaler(img: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = img.mean()
    var = img.var()
    return (img - mean) / np.sqrt(var + eps)
