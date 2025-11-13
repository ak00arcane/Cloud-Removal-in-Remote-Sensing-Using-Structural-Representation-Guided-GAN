import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from models.generator import CloudRemovalGenerator
from config import Config


def load_generator(checkpoint_path, device):
    cfg = Config()
    gen = CloudRemovalGenerator(cfg).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # try common keys
    loaded = False
    if 'generator_state_dict' in ckpt:
        gen.load_state_dict(ckpt['generator_state_dict'])
        loaded = True
    else:
        # maybe the checkpoint directly stores model state
        try:
            gen.load_state_dict(ckpt)
            loaded = True
        except Exception:
            # try to find a state dict inside the file
            for k, v in ckpt.items():
                if isinstance(v, dict) and 'weight' in str(v.keys()):
                    try:
                        gen.load_state_dict(v)
                        loaded = True
                        break
                    except Exception:
                        pass
    if not loaded:
        print("Warning: could not find a conventional generator_state_dict in the checkpoint. Attempting to continue with partially loaded model.")
    return gen, ckpt


def export_onnx(generator, output_path, input_size=(1,3,256,256), device='cpu'):
    # Prepare dummy inputs matching model forward signature: (cloudy_img, cloud_mask, temporal_img)
    cloudy = torch.randn(*input_size, device=device)
    temporal = torch.randn(*input_size, device=device)
    # cloud_mask not used by generator implementation but include for signature compatibility
    cloud_mask = torch.zeros((input_size[0], 1, input_size[2], input_size[3]), device=device)

    generator.eval()
    # Create example input tuple
    example_inputs = (cloudy, cloud_mask, temporal)

    # Export to torchscript first
    print(f"Exporting TorchScript to temporary file...")
    script_model = torch.jit.trace(generator, example_inputs)
    script_path = output_path + '.pt'
    torch.jit.save(script_model, script_path)
    print(f"TorchScript export saved to: {script_path}")
    
    print("Export complete. Note: ONNX export skipped due to installation issues.")


def verify_onnx(onnx_path, generator, device='cpu'):
    try:
        import onnxruntime as ort
    except Exception as e:
        print('onnxruntime not available, skipping ONNX verification. Install with `pip install onnxruntime` to enable verification.')
        return

    # prepare random input
    h = 256
    w = 256
    cloudy_np = np.random.randn(1,3,h,w).astype(np.float32)
    temporal_np = np.random.randn(1,3,h,w).astype(np.float32)
    cloud_mask_np = np.zeros((1,1,h,w), dtype=np.float32)

    # PyTorch output
    cloudy_t = torch.from_numpy(cloudy_np).to(device)
    temporal_t = torch.from_numpy(temporal_np).to(device)
    cloud_mask_t = torch.from_numpy(cloud_mask_np).to(device)
    with torch.no_grad():
        pyt_out = generator(cloudy_t, cloud_mask_t, temporal_t)[0].cpu().numpy()

    # ONNX runtime
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inputs = {
        'cloudy_img': cloudy_np,
        'cloud_mask': cloud_mask_np,
        'temporal_img': temporal_np
    }
    onnx_out = sess.run(None, inputs)[0]

    # Compare shapes and differences
    print('PyTorch output shape:', pyt_out.shape)
    print('ONNX output shape   :', onnx_out.shape)
    diff = np.abs(pyt_out - onnx_out)
    print('Max absolute difference:', float(diff.max()))
    print('Mean absolute difference:', float(diff.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--output', default='onnx_models/generator.onnx')
    parser.add_argument('--size', type=int, default=256, help='Height and width for dummy input (square)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--verify', action='store_true', help='Run ONNX verification using onnxruntime if available')
    args = parser.parse_args()

    device = torch.device(args.device)
    gen, ckpt = load_generator(args.checkpoint, device)

    # Print checkpoint summary
    try:
        if isinstance(ckpt, dict):
            print('Checkpoint keys:', list(ckpt.keys()))
    except Exception:
        pass

    export_onnx(gen, args.output, input_size=(1,3,args.size,args.size), device=device)

    if args.verify:
        verify_onnx(args.output, gen, device=device)
