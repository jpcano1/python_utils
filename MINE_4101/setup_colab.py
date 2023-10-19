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


def setup_lab5():
    setup_general.setup_general()
    from utils import general as gen

    id_insurance = "11jaAMuHLypta8BXyUfPPOnql-PXzLQGD"
    gen.download_file_from_google_drive(id_insurance, "insurance.csv")
    id_wine = "1Je03icLBNGad8q58QnJ-eQKex82t3exP"
    gen.download_file_from_google_drive(id_wine, "winequality.csv")


def setup_lab6():
    setup_general.setup_general()
    from utils import general as gen

    url_diet = (
        "https://github.com/hctorresm/Monitoria_CDA/blob/master/Clase9_Python/Diet.xlsx?raw=true"
    )
    gen.download_content(url_diet, "Diet.xlsx")
    print("Lab 6 enabled successfully")


def setup_project(filename: str = "kaggle.json", download_dataset=True):
    setup_general.setup_general()
    setup_kaggle_token(filename)
    os.system("pip install -q kaggle==1.5.6")
    if download_dataset:
        os.system("kaggle competitions download -c house-prices-advanced-regression-techniques")
        from utils import general as gen

        gen.extract_file("house-prices-advanced-regression-techniques.zip", "data")
    print("Data Science Project Enabled Successfully")
