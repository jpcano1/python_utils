import setup_colab_general as setup_general
import os
from google.colab import files
from IPython.display import clear_output

def setup_kaggle_token(filename: str):
    assert filename.endswith(".json"), "El archivo no es JSON"
    files.upload()
    clear_output(wait=True)
    os.system("mkdir ~/.kaggle")
    os.system(f"cp {filename} ~/.kaggle/")
    os.system(f"chmod 600 ~/.kaggle/{filename}")

def setup_project(filename="kaggle.json", download_dataset=True, 
                  kaggle_version="1.5.6"):
    setup_general.setup_general()
    setup_kaggle_token(filename)
    os.system(f"pip install -q kaggle=={kaggle_version}")
    if download_dataset:
        os.system("kaggle datasets download -d tourist55/alzheimers-dataset-4-class-of-images")
        from utils import general as gen
        gen.extract_file("alzheimers-dataset-4-class-of-images.zip", "data")
        print("Dataset Downloaded Successfully")
    print("Workshop Project Enabled Successfully")
