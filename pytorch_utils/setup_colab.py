import setup_colab_general as setup_general
import os

def setup_architecture(architecture="unet"):
    assert architecture in ["unet", "runet"]

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
    if architecture == "unet":
        unet_path = "pytorch_utils/unet_architectures/unet.py"
        setup_general.download_github_content(unet_path, "utils/unet.py")
        print("U-Net Enabled")
    if architecture == "runet":
        runet_path = "pytorch_utils/unet_architectures/runet.py"
        setup_general.download_github_content(runet_path, "utils/runet.py")
        print("RU-Net Enabled")