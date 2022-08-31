import os

import torch
from timm.models import create_model
from datasets import build_dataset
import argparse
from tqdm import tqdm
import models
import models_v2
from extractor_deit import VitExtractor_deit

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment
    parser.add_argument('--default_transform', action='store_true')  # 3augment
    parser.set_defaults(default_transform=False)
    parser.add_argument('--src', action='store_true')  # simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/dataset', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    return parser


def save_psuedo_label(model, data_loader, save_root,device):
    idx = 0
    for batch in tqdm(data_loader):
        image,label,name = batch
        image = image.to(device)
        svae_path = save_root+"/"+str(idx).zfill(8)
        with torch.no_grad():
            feature_list = model.get_feature_from_input(image)
            feature_all = torch.stack(feature_list,dim=1)
            feature_all = feature_all[:,:,0,:].detach().cpu()
        #save all the dict
        torch.save({"img":feature_all,
                    "label":label,
                    "name":name},svae_path)
        idx+=1



def main():

    model_name = "deit_tiny_patch16_224"
    nb_classes = 1000
    drop = 0.
    drop_path = 0.
    input_size = 224

    model = create_model(
        model_name,
        pretrained=False,
        num_classes=nb_classes,
        drop_rate=drop,
        drop_path_rate=drop_path,
        drop_block_rate=None,
        img_size=input_size
    )

    # ckpt_path = "./training/train_val_noaug/checkpoint.pth"
    ckpt_path = "./training/train_noaug/checkpoint.pth"
    device = torch.device("cuda")
    state_dict = torch.load(ckpt_path,map_location="cpu")
    model.load_state_dict(state_dict['model'])
    model.to(device)
    extract = VitExtractor_deit(model)

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    train_dataset,_ = build_dataset(is_train=True, args=args)
    val_dataset,_ = build_dataset(is_train=False, args=args)

    num_workers = 10
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    # root_path = "./dataset_feature/train_val_noaug"
    root_path = "./dataset_feature/train_noaug"
    train_f_path = root_path+"/train_ori"
    val_f_path = root_path+"/val_ori"

    os.makedirs(train_f_path,exist_ok=True)
    os.makedirs(val_f_path, exist_ok=True)
    #compute and save the features, image path and label
    save_psuedo_label(extract,data_loader_train,train_f_path,device)
    save_psuedo_label(extract, data_loader_val, val_f_path, device)


if __name__=="__main__":
    main()