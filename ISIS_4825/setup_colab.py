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

def setup_workshop_8():
    setup_general.setup_general()
    print("Workshop 8 Enabled Successfully")

def setup_workshop_9(filename: str="kaggle.json", download_dataset=True):
    setup_general.setup_general()
    setup_kaggle_token(filename)
    os.system("pip install -q kaggle==1.5.6")
    if download_dataset:
        os.system("kaggle datasets download -d datamunge/sign-language-mnist")
        from utils import general as gen
        gen.extract_file("sign-language-mnist.zip", "data")
    print("Workshop 9 Enabled Successfully")

def setup_workshop_12(download_dataset=True):
    setup_general.setup_general()
    torch_path = "ISIS_4825/ML/Taller_12/torch_utils.py"
    setup_general.download_github_content(torch_path, "utils/torch_utils.py")
    from utils import general as gen
    if download_dataset:
        train_id = "1qRwpSLj6POrJV7ZJFTPt0kswwrxtDA8b"
        test_id = "1--mp95lbSA-EiElFIR_2xTbylsARsoM2"
        gen.download_file_from_google_drive(train_id, "train_data.zip", size=209.1e3, zip=True)
        gen.download_file_from_google_drive(test_id, "test_data.zip", size=11e3, zip=True)
    print("Workshop 12 Enabled Successfully")