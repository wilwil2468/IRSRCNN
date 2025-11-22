"""
Demo script for quantized SRCNN model - single image super-resolution
FIXED: Uses RGB color space (matching training) instead of YCbCr
"""
import torch
from quantized_model import SRCNN_Quantized
import argparse

# Import common utilities if available
try:
    from utils.common import (
        read_image, write_image, gaussian_blur, upscale, norm01, denorm01
    )
    USE_COMMON = True
except ImportError:
    USE_COMMON = False
    print("Warning: utils.common not found, using fallback implementations")

# Fallback implementations
if not USE_COMMON:
    import torchvision.io as io
    import torchvision.transforms as transforms
    
    def read_image(filepath):
        image = io.read_image(filepath, io.ImageReadMode.RGB)
        return image
    
    def write_image(filepath, src):
        io.write_png(src, filepath)
    
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
    
    def denorm01(src):
        return src * 255


# -----------------------------------------------------------
# argument parser
# -----------------------------------------------------------

parser = argparse.ArgumentParser(description='SRCNN Quantized Demo - Single Image Super-Resolution')
parser.add_argument('--scale', type=int, default=2, help='Scale factor (2, 3, or 4)')
parser.add_argument('--ckpt-path', type=str, default="", help='Path to model checkpoint')
parser.add_argument('--architecture', type=str, default="955", help='Architecture (915, 935, or 955)')
parser.add_argument('--bit-width', type=int, default=8, help='Quantization bit width')
parser.add_argument('--image-path', type=str, default="dataset/test1.png", help='Input image path')
parser.add_argument('--output-bicubic', type=str, default="bicubic.png", help='Bicubic output path')
parser.add_argument('--output-sr', type=str, default="sr_quantized.png", help='SRCNN output path')

FLAGS, unparsed = parser.parse_known_args()

image_path = FLAGS.image_path
architecture = FLAGS.architecture
scale = FLAGS.scale
bit_width = FLAGS.bit_width
output_bicubic = FLAGS.output_bicubic
output_sr = FLAGS.output_sr

# Validation
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, or 955")

if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

# Set checkpoint path
ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"weights/SRCNN-{architecture}-w{bit_width}-flir-best.pt"

# Preprocessing parameters
sigma = 0.3 if scale == 2 else 0.2
pad = int(architecture[1]) // 2 + 6


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    import os
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        print("Please provide a valid checkpoint path with --ckpt-path")
        return
    
    # Check if input image exists
    if not os.path.exists(image_path):
        print(f"Error: Input image not found: {image_path}")
        print("Please provide a valid image path with --image-path")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {ckpt_path}")
    print(f"Architecture: {architecture}, Bit-width: {bit_width}, Scale: {scale}")
    print(f"Input image: {image_path}")
    print(f"Color space: RGB (matching training)")
    print("=" * 70)
    
    # Load and preprocess image for bicubic baseline
    print("Processing bicubic upscaling...")
    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = bicubic_image[:, pad:-pad, pad:-pad]
    write_image(output_bicubic, bicubic_image)
    print(f"✓ Bicubic result saved to: {output_bicubic}")
    
    # Load and preprocess image for SRCNN
    print("\nProcessing SRCNN super-resolution...")
    lr_image = read_image(image_path)
    lr_image = gaussian_blur(lr_image, sigma=sigma)
    bicubic_image = upscale(lr_image, scale)
    
    # STAY IN RGB - no color space conversion
    bicubic_image = norm01(bicubic_image)
    bicubic_image = torch.unsqueeze(bicubic_image, dim=0)
    
    # Load quantized model
    model = SRCNN_Quantized(architecture=architecture, bit_width=bit_width)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        bicubic_image = bicubic_image.to(device)
        sr_image = model(bicubic_image)[0].cpu()
    
    # Postprocess and save - STAY IN RGB
    sr_image = denorm01(sr_image)
    sr_image = sr_image.type(torch.uint8)
    
    write_image(output_sr, sr_image)
    print(f"✓ SRCNN result saved to: {output_sr}")
    
    print("=" * 70)
    print("Demo complete!")
    print(f"  Bicubic output: {output_bicubic}")
    print(f"  SRCNN output:   {output_sr}")
    print("=" * 70)


if __name__ == "__main__":
    main()