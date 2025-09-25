import os
import subprocess
import sys

def setup_environment():
    # Set environment variable
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Install packages
    packages = [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "tensorboard",
        "scikit-learn",
    ]

    print("Installing packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

    print("Setup complete. You can now run TensorBoard with:")
    print("    from torch.utils.tensorboard import SummaryWriter")
    print("    # or launch: tensorboard --logdir=runs")
