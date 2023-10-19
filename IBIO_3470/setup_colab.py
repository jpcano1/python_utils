import setup_colab_general as setup_general

from IPython.display import clear_output
from google.colab import files
import os


def setup_kaggle_token(filename: str):
    assert filename.endswith(".json"), "El archivo no es JSON"
    files.upload()
    clear_output(wait=True)
    os.system("mkdir ~/.kaggle")
    os.system(f"cp {filename} ~/.kaggle/")
    os.system(f"chmod 600 ~/.kaggle/{filename}")


def setup_project(download_dataset=True):
    setup_general.setup_general(dst="libs")
    if download_dataset:
        from libs import general as gen

        data_id = "1VZ9MvahloAaejUZUci-q_CIHTzfFtzsK"
        utils_url = "https://raw.githubusercontent.com/jpcano1/python_utils/main/IBIO_3470/final_project/utils.py"
        gen.download_file_from_google_drive(data_id, "data.zip", size=35.87e3)
        gen.download_content(utils_url, "utils.py", "libs")
        print("Dataset Downloaded Successfully")
    print("Workshop Project Enabled Successfully")
