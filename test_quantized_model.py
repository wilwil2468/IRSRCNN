"""
Test quantized SRCNN model on FLIR thermal images
FIXED: Uses RGB color space (matching training) instead of YCbCr
"""
import torch
import torch.nn.functional as F
from quantized_model import SRCNN_Quantized
import torchvision.io as io
import torchvision.transforms as transforms
import argparse

# Import common utilities if available
try:
    from utils.common import (
        read_image, gaussian_blur, upscale, 
        norm01, PSNR, sorted_list
    )
    USE_COMMON = True
except ImportError:
    USE_COMMON = False
    print("Warning: utils.common not found, using fallback implementations")


# Fallback implementations if utils.common is not available
def read_image(filepath):
    image = io.read_image(filepath, io.ImageReadMode.RGB)
    return image


def gaussian_blur(src, ksize=3, sigma=0.5):
    blur_image = transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)(src)
    return blur_image


def upscale(src, scale):
    h = int(src.shape[1] * scale)
    w = int(src.shape[2] * scale)
    image = transforms.Resize((h, w), transforms.InterpolationMode.BICUBIC)(src)
    return image


def norm01(src):
    return src / 255


def PSNR(y_true, y_pred, max_val=1):
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)
    MSE = torch.mean(torch.square(y_true - y_pred))
    return 10 * torch.log10(max_val * max_val / MSE)


def sorted_list(dir):
    import os
    ls = os.listdir(dir)
    ls.sort()
    for i in range(0, len(ls)):
        ls[i] = os.path.join(dir, ls[i])
    return ls


# -----------------------------------------------------------
# argument parser
# -----------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=2, help='Scale factor (2, 3, or 4)')
parser.add_argument('--architecture', type=str, default="955", help='Architecture (915, 935, or 955)')
parser.add_argument('--bit-width', type=int, default=8, help='Quantization bit width')
parser.add_argument('--ckpt-path', type=str, default="", help='Path to model checkpoint')
parser.add_argument('--data-dir', type=str, default="dataset/test_ir/x2/data", help='LR images directory')
parser.add_argument('--labels-dir', type=str, default="dataset/test_ir/x2/labels", help='HR images directory')

FLAGS, unparsed = parser.parse_known_args()

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, 955")

bit_width = FLAGS.bit_width

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"weights/SRCNN-{architecture}-w{bit_width}-flir-best.pt"

data_dir = FLAGS.data_dir
labels_dir = FLAGS.labels_dir

# Sigma and padding based on scale
sigma = 0.3 if scale == 2 else 0.2
pad = int(architecture[1]) // 2 + 6


# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

def main():
    import os
    
    # Check paths exist
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint not found: {ckpt_path}")
        print("Please provide a valid checkpoint path with --ckpt-path")
        return
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please provide a valid data directory with --data-dir")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        print("Please provide a valid labels directory with --labels-dir")
        return
    
    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {ckpt_path}")
    print(f"Architecture: {architecture}, Bit-width: {bit_width}, Scale: {scale}")
    print(f"Color space: RGB (matching training)")
    
    model = SRCNN_Quantized(architecture=architecture, bit_width=bit_width)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get image lists
    ls_data = sorted_list(data_dir)
    ls_labels = sorted_list(labels_dir)
    
    if len(ls_data) == 0:
        print(f"Error: No images found in {data_dir}")
        return
    
    if len(ls_data) != len(ls_labels):
        print(f"Warning: Number of data images ({len(ls_data)}) != number of label images ({len(ls_labels)})")
        print(f"Using minimum count: {min(len(ls_data), len(ls_labels))}")
    
    num_images = min(len(ls_data), len(ls_labels))
    print(f"Testing on {num_images} images...")
    print("=" * 70)
    
    sum_psnr = 0
    with torch.no_grad():
        for i in range(num_images):
            # Read LR image and apply preprocessing
            lr_image = read_image(ls_data[i])
            lr_image = gaussian_blur(lr_image, sigma=sigma)
            bicubic_image = upscale(lr_image, scale)
            
            # Read HR image
            hr_image = read_image(ls_labels[i])
            
            # Crop HR image (account for padding)
            hr_image = hr_image[:, pad:-pad, pad:-pad]
            
            # Normalize to [0, 1] - STAY IN RGB (no YCbCr conversion)
            bicubic_image = norm01(bicubic_image)
            hr_image = norm01(hr_image)
            
            # Run inference
            bicubic_image = torch.unsqueeze(bicubic_image, dim=0).to(device)
            sr_image = model(bicubic_image)[0].cpu()
            
            # Crop both hr and sr to the overlapping valid region
            # Ensure same HxW before PSNR calculation
            _, H_hr, W_hr = hr_image.shape
            _, H_sr, W_sr = sr_image.shape
            H = min(H_hr, H_sr)
            W = min(W_hr, W_sr)
            hr_cropped = hr_image[:, :H, :W]
            sr_cropped = sr_image[:, :H, :W]
            
            # Calculate PSNR
            psnr = PSNR(hr_cropped, sr_cropped, max_val=1)
            sum_psnr += psnr
            
            # Print progress every 10 images
            if (i + 1) % 10 == 0 or (i + 1) == num_images:
                print(f"Processed {i + 1}/{num_images} images...")
    
    # Print average PSNR
    avg_psnr = sum_psnr.numpy() / num_images
    print("=" * 70)
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print("=" * 70)


if __name__ == "__main__":
    main()