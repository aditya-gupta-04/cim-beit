# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on DINO code bases
# https://github.com/facebookresearch/dino/blob/main/eval_linear.py
# --------------------------------------------------------'
import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import modeling_pretrain
from datasets import build_beit_pretraining_dataset
from timm.models import create_model
from typing import Iterable
from tqdm import tqdm

import matplotlib.pyplot as plt

def load_model(model, checkpoint_file, model_key, model_prefix):
    if checkpoint_file.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_file, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

    checkpoint_model = None
    for model_key in model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint

    utils.load_state_dict(model, checkpoint_model, prefix=model_prefix)


def eval_linear(args):
    cudnn.benchmark = True

    mean = (0.485, 0.456, 0.406) if args.imagenet_default_mean_and_std else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if args.imagenet_default_mean_and_std else (0.5, 0.5, 0.5)

    # ============ building network ... ============
    model = create_model(
        args.model, pretrained=False, num_classes=0, drop_rate=0, drop_path_rate=0,
        attn_drop_rate=0, drop_block_rate=None, use_mean_pooling=False,
        use_shared_rel_pos_bias=args.rel_pos_bias, use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    patch_size = model.patch_embed.patch_size
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])

    # ============ preparing data ... ============
    
    dataset = build_beit_pretraining_dataset(args)
    sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    device = args.device
    model.to(device)
    model.eval()
    print(f"Model {args.model} built.")

    # load weights to evaluate
    load_model(model=model, checkpoint_file=args.pretrained_weights, model_key=args.checkpoint_key, model_prefix="")
    d_vae = utils.create_d_vae(
        weight_path=args.discrete_vae_weight_path, d_vae_type=args.discrete_vae_type,
        device=device, image_size=args.second_input_size)

    validate_one_epoch(model, d_vae, data_loader, device)

def validate_one_epoch(model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, device: torch.device):
    model.eval()

    for step, (batch, _) in tqdm(enumerate(data_loader), total=len(data_loader)):

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        with torch.no_grad():
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            input_ids = d_vae.get_codebook_indices(images).flatten(1)
            true_tokens = input_ids[~bool_masked_pos]

            with torch.cuda.amp.autocast():
                outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False)

                corrupted_img_tokens = torch.zeros(bool_masked_pos.flatten().shape).cuda().to(true_tokens.dtype)
                corrupted_img_tokens[bool_masked_pos.flatten()] = torch.argmax(outputs, axis=1)
                corrupted_img_tokens[~bool_masked_pos.flatten()] = true_tokens

                corrupted_images = d_vae.decode(corrupted_img_tokens.view(images.shape[0], -1))
                
                for i in range(images.shape[0]):
                    plt.subplot(1,3,1)
                    plt.imshow(samples[i].detach().cpu().numpy().transpose(1,2,0))
                    plt.subplot(1,3,2)
                    plt.imshow(images[i].detach().cpu().numpy().transpose(1,2,0))
                    plt.subplot(1,3,3)
                    plt.imshow(corrupted_images[i].detach().cpu().numpy().transpose(1,2,0)[:,:,:3])
                    plt.savefig(f"results/{step*images.shape[0] + i}.png")



def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--avgpool_patchtokens', default=True, type=bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to True for BEiT pretrained models. """)
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--optimizer', default="adamw", type=str, help='optimizer type')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="model|module|teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument("--base_lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, type=bool_flag,
        help="""Set True to use the imagenet default mean and std, Set False will use the mean and std in Inception. 
        We recommand keep it same to the pre-training stage. """)
    parser.add_argument('--amp_forward', default=True, type=bool_flag, help='Use amp to inference the pre-trained model, which can speed up the evaluation. ')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--discrete_vae_weight_path", type=str)
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e")

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=112, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')
    
    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
