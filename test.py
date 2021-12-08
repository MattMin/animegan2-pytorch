import argparse
import os
import re

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def test(args):
    device = args.device
    for checkpoint in args.checkpoint.split(";"):
        net = Generator()
        net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        net.to(device).eval()
        print(f"model loaded: {checkpoint}")

        os.makedirs(args.output_dir, exist_ok=True)
        weight = re.findall("./weights/(.+?).pt", checkpoint)
        for image_name in sorted(os.listdir(args.input_dir)):
            if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
                continue

            image = load_image(os.path.join(args.input_dir, image_name), args.x32)

            with torch.no_grad():
                image = to_tensor(image).unsqueeze(0) * 2 - 1
                out = net(image.to(device), args.upsample_align).cpu()
                out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
                out = to_pil_image(out)
            new_name = weight[0] + "_" + image_name
            out.save(os.path.join(args.output_dir, new_name))
            print(f"image saved: {new_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./weights/celeba_distill.pt;./weights/face_paint_512_v1.pt;./weights/face_paint_512_v2.pt;./weights/paprika.pt',
        # default='./weights/face_paint_512_v1.pt',
        # default='./weights/face_paint_512_v2.pt',
        # default='./weights/paprika.pt',
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./samples/results',
    )
    parser.add_argument(
        '--device',
        type=str,
        # default='cuda:0',
        default='cpu',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
        help="Align corners in decoder upsampling layers"
    )
    parser.add_argument(
        '--x32',
        action="store_true",
        help="Resize images to multiple of 32"
    )
    args = parser.parse_args()

    test(args)
