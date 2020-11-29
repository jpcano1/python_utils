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

def setup_workshop_9(filename: str="kaggle.json", 
                    download_dataset=True, kaggle_version="1.5.6"):
    setup_general.setup_general()
    setup_kaggle_token(filename)
    os.system(f"pip install -q kaggle=={kaggle_version}")
    if download_dataset:
        os.system("kaggle datasets download -d datamunge/sign-language-mnist")
        from utils import general as gen
        gen.extract_file("sign-language-mnist.zip", "data")
        print("Dataset Downloaded Successfully")
    print("Workshop 9 Enabled Successfully")

def setup_workshop_10(download_dataset=True):
    setup_general.setup_general()
    if download_dataset:
        from utils import general as gen
        data_url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        gen.download_content(data_url, "data.tar.gz")
        print("Dataset Downloaded Successfully")
    print("Workshop 10 Enabled Successfully")

def setup_workshop_12(pretrained=True, download_dataset=True):
    setup_general.setup_general()
    tf_utils_path = "ISIS_4825/ML/Taller_12/tf_utils.py"
    setup_general.download_github_content(tf_utils_path, "utils/tf_utils.py")
    print("Util Functions Downloaded Successfully")
    from utils import general as gen
    if download_dataset:
        data_id = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"
        labels_url = "https://github.com/AlexOlsen/DeepWeeds/raw/master/labels/labels.csv"
        gen.download_file_from_google_drive(data_id, "images.zip", size=491e3)
        gen.download_content(labels_url, "labels.csv", "data")
        print("Dataset Downloaded Successfully")
    if pretrained:
        inception_id = "1Us6XwnpyWYCCY2wRzBTr9ep6kRXcL-jz"
        gen.download_file_from_google_drive(inception_id, "inception.h5", dst="models", size=89e3)
        print("Pretrained Networks Downloaded Successfully")
    print("Workshop 12 Enabled Successfully")

def setup_workshop_13(download_dataset=True, pretrained=True, 
                      brats=False):
    setup_general.setup_general()
    os.system("pip install -q albumentations==0.5.0")
    torch_path = "ISIS_4825/ML/Taller_13/torch_utils.py"
    vis_path = "ISIS_4825/ML/Taller_13/visualization_utils.py"
    layers_path = "ISIS_4825/ML/Taller_13/layers.py"
    train_path = "ISIS_4825/ML/Taller_13/train_utils.py"
    setup_general.download_github_content(torch_path, "utils/torch_utils.py")
    setup_general.download_github_content(layers_path, "utils/layers.py")
    setup_general.download_github_content(train_path, "utils/train_utils.py")
    setup_general.download_github_content(vis_path, "utils/visualization_utils.py")
    print("Util Functions Downloaded Successfully")
    from utils import general as gen
    if download_dataset:
        train_id = "192V5FfehmbpN2wkl1apiygxSqW6EUmyP"
        test_id = "1--hE7Ucvlsjf-fwET4-JMuS-VAva7Vxq"
        gen.download_file_from_google_drive(train_id, "train_data.zip", size=212e3)
        gen.download_file_from_google_drive(test_id, "test_data.zip", size=22e3)
        print("Dataset Downloaded Successfully")
    if pretrained:
        autoencoder_id = "1ZGDKYCHZFpW1D-rx8xmxxjJiFoc5N1Au"
        unet_id = "1kfd4_30DIdp3Di8fU5_h4I-EdFKEY7Uq"
        gen.download_file_from_google_drive(autoencoder_id, "autoencoder.pt",
                                            dst="./models", size=49e3)
        gen.download_file_from_google_drive(unet_id, "unet.pt",
                                            dst="./models", size=22e3)
        print("Pretrained Networks Downloaded Successfully")
    if brats:
        part_1 = "1NRAFehzp6WtpoQduMJ5PGt1ivC8NhpJ1"
        part_2 = "1-6vfu9nBu1PIabcZPMAmX685M1mOkceI"
        gen.download_file_from_google_drive(part_1, 
                                            "brats_part_1.zip", 
                                            size=426.8e3)
        gen.download_file_from_google_drive(part_2,
                                            "brats_part_2.zip",
                                            size=370.3e3)
        print("Dataset BraTS Downloaded Successfully")
    print("Workshop 13 Enabled Successfully")

def setup_extra_workshop(download_dataset=True):
    setup_general.setup_general()
    extra_utils_path = "ISIS_4825/ML/Taller_Extra/extra_utils.py"
    setup_general.download_github_content(extra_utils_path, "utils/extra_utils.py")
    print("Util Functions Downloaded Successfully")
    if download_dataset:
        from utils import general as gen
        id_data = "0B0vscETPGI1-TE5KWFgxaURubFE"
        gen.download_file_from_google_drive(id_data, "kits.zip", size=4.27e6)
        print("Dataset Downloaded")
    print("Extra Workshop Enabled Successfully")