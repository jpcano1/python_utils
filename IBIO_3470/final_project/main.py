import argparse

from libs.trainer import evaluate
from libs import general as gen

import os

def setup_project():
    if os.path.exists("data") and os.path.exists("models"):
        return
    data_id = "1VZ9MvahloAaejUZUci-q_CIHTzfFtzsK"
    gen.download_file_from_google_drive(data_id, "data.zip", size=35.87e3)
    model_id = "1qrq9SRWasvsi7_UMFRnzYljDbM0XFSqL"
    gen.download_file_from_google_drive(model_id, "model.npy", 
                                        dst="models", size=986e3)
    print("Dataset Downloaded Successfully")
    print("Workshop Project Enabled Successfully")

if __name__ == "__main__":
    setup_project()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image", type=str, default=None,
        help="Nombre de la imagen que se va a evaluar"
    )

    args = parser.parse_args()

    evaluate(image=args.image)
