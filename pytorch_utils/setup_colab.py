import setup_colab_general as setup_general

def setup_unet():
    setup_general.setup_general()
    general_layers_path = "pytorch_utils/general_layers.py"
    unet_path = "pytorch_utils/unet_architectures/unet.py"
    setup_general.download_github_content(general_layers_path, "utils/general_layers.py")
    setup_general.download_github_content(unet_path, "utils/unet.py")
    print("U-Net Functions Enabled!")

def setup_runet():
    setup_general.setup_general()
    general_layers_path = "pytorch_utils/general_layers.py"
    layers_path = "pytorch_utils/unet_architectures/layers.py"
    runet_path = "pytorch_utils/unet_architectures/runet.py"
    setup_general.download_github_content(general_layers_path, "utils/general_layers.py")
    setup_general.download_github_content(layers_path, "utils/layers.py")
    setup_general.download_github_content(runet_path, "utils/runet.py")
    print("RU-Net Functions Enabled")