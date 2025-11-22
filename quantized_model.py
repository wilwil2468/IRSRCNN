"""
Quantized SRCNN model using Brevitas for FINN deployment
This requires: pip install brevitas
"""
import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

class SRCNN_Quantized(nn.Module):
    """
    Quantized SRCNN model compatible with FINN framework
    """
    def __init__(self, architecture="955", bit_width=8):
        super(SRCNN_Quantized, self).__init__()

        if architecture not in ["915", "935", "955"]:
            raise ValueError("architecture must be 915, 935 or 955")
        k = int(architecture[1])

        # Input quantization
        self.input_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Patch extraction layer
        self.patch_extraction = QuantConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=9,
            padding=4,  # (9-1)/2 = 4 to maintain spatial dimensions
            weight_bit_width=bit_width,
            bias=True,
            return_quant_tensor=True
        )
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Non-linear mapping layer
        # For 955: k=5, padding=2; for 935: k=3, padding=1; for 915: k=1, padding=0
        padding_map = {1: 0, 3: 1, 5: 2}
        self.nonlinear_map = QuantConv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=k,
            padding=padding_map[k],  # Maintain spatial dimensions
            weight_bit_width=bit_width,
            bias=True,
            return_quant_tensor=True
        )
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Reconstruction layer
        self.recon = QuantConv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=5,
            padding=2,  # (5-1)/2 = 2 to maintain spatial dimensions
            weight_bit_width=bit_width,
            bias=True,
            return_quant_tensor=False  # Final layer outputs float
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.patch_extraction(x)
        x = self.relu1(x)
        x = self.nonlinear_map(x)
        x = self.relu2(x)
        x = self.recon(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x


def transfer_weights_from_pretrained(quantized_model, pretrained_weights_path):
    """
    Transfer weights from float32 model to quantized model
    This provides a good starting point for fine-tuning

    Args:
        quantized_model: Brevitas quantized model
        pretrained_weights_path: Path to original float32 weights
    """
    import sys
    import os

    # Ensure we import from local neuralnet.py, not system package
    # Add current directory to path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Import from local neuralnet.py file
    from neuralnet import SRCNN_model

    # Load pretrained model
    architecture = "955"  # Adjust based on your model
    pretrained_model = SRCNN_model(architecture)
    pretrained_model.load_state_dict(torch.load(pretrained_weights_path, map_location='cpu'))

    # Transfer weights (Brevitas layers have .weight and .bias attributes)
    with torch.no_grad():
        # Patch extraction
        quantized_model.patch_extraction.weight.copy_(pretrained_model.patch_extraction.weight)
        quantized_model.patch_extraction.bias.copy_(pretrained_model.patch_extraction.bias)

        # Nonlinear mapping
        quantized_model.nonlinear_map.weight.copy_(pretrained_model.nonlinear_map.weight)
        quantized_model.nonlinear_map.bias.copy_(pretrained_model.nonlinear_map.bias)

        # Reconstruction
        quantized_model.recon.weight.copy_(pretrained_model.recon.weight)
        quantized_model.recon.bias.copy_(pretrained_model.recon.bias)

    print("Weights transferred successfully!")
    print("Note: You should fine-tune this model with quantization-aware training")

    return quantized_model


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Quantized SRCNN Model Example")
    print("=" * 60)

    # Create quantized model
    model = SRCNN_Quantized(architecture="955", bit_width=8)
    print("✓ Quantized model created")

    # Transfer pretrained weights (optional)
    import os
    if os.path.exists("weights/SRCNN-955.pt"):
        model = transfer_weights_from_pretrained(
            model,
            "weights/SRCNN-955.pt"
        )
        print("✓ Pretrained weights transferred")
    else:
        print("⚠ No pretrained weights found, skipping transfer")
        print("  You can train this model from scratch or provide pretrained weights")

    # Export to QONNX for FINN
    from brevitas.export import export_qonnx
    dummy_input = torch.randn(1, 3, 224, 224)

    output_path = "weights/SRCNN-955-quantized.qonnx"
    os.makedirs("weights", exist_ok=True)

    export_qonnx(
        model,
        dummy_input,
        export_path=output_path
    )
    print(f"✓ QONNX model exported to: {output_path}")
    print("\nNext steps:")
    print("1. Fine-tune with train_quantized.py for best accuracy")
    print("2. Use FINN to generate FPGA bitstream")
    print("3. Deploy to PYNQ-Z2 board")
    print("=" * 60)
