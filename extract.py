import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True, help="Epoch number to process")
    return parser.parse_args()


args = parse_args()
epoch = args.epoch

ckpt_file_name = f"epoch={epoch}.ckpt"

ckpt_path = os.path.join(f"./experiments/temos/H3D-TMR-v1/checkpoints/{ckpt_file_name}")
extracted_path = os.path.join("./experiments/temos/H3D-TMR-v1/extract_weights")
os.makedirs(extracted_path, exist_ok=True)

new_path_template = os.path.join(extracted_path, "{}.pt")
ckpt_dict = torch.load(ckpt_path)
state_dict = ckpt_dict["state_dict"]

for modal in ["motion","text"]:
    path = os.path.join(extracted_path, f"{modal}_{ckpt_file_name}")
    module_name = f"{modal}encoder"
    sub_state_dict = {
        ".".join(x.split(".")[1:]): y.cpu()
        for x, y in state_dict.items()
        if x.split(".")[0] == module_name
    }
    torch.save(sub_state_dict, path)
