import setup_colab_general as setup_general
import os

def setup_architecture(architecture="unet"):
    os.makedirs("train_utils", exist_ok=True)
    with open("train_utils/__init__.py", "wb") as f:
        f.close()
    setup_general.setup_general()
    metrics_path = "pytorch_utils/train_utils/metrics.py"
    train_loop_path = "pytorch_utils/train_utils/seg_train_loops.py"
    layers_path = "pytorch_utils/unet_architectures/layers.py"
    general_layers_path = "pytorch_utils/general_layers.py"
    setup_general.download_github_content(general_layers_path, 
                                          "utils/general_layers.py")
    setup_general.download_github_content(metrics_path,
                                          "train_utils/metrics.py")
    setup_general.download_github_content(train_loop_path, 
                                          "train_utils/train_loop.py")
    setup_general.download_github_content(layers_path, "utils/layers.py")
    print("Layers and utils enabled")
    
    if architecture == "unet" or "unet" in architecture:
        unet_path = "pytorch_utils/unet_architectures/unet.py"
        setup_general.download_github_content(unet_path, "utils/unet.py")
        print("U-Net Enabled")
    if architecture == "runet" or "runet" in architecture:
        runet_path = "pytorch_utils/unet_architectures/runet.py"
        setup_general.download_github_content(runet_path, "utils/runet.py")
        print("RU-Net Enabled")

    if architecture == "r2unet" or "r2unet" in architecture:
        r2unet_path = "pytorch_utils/unet_architectures/runet.py"
        setup_general.download_github_content(r2unet_path, "utils/r2unet.py")
        print("R2U-Net Enabled")