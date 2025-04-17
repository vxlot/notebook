"""
Pick-a-Pic Dataset Loader Script (2025.4.13 by user 110003)

Description:
    This script loads the Pick-a-Pic v2 dataset from Hugging Face or local path,
    by the way, change the `split` list in `dataset_info.json` can shrink the size of it.

Might set if you want to download a new version of the dataset:
    - export HF_ENDPOINT=https://hf-mirror.com
    - export HF_HOME="your local cache directory" if you dont't specify the `cache_dir`
"""

import argparse
import io
import random
from datasets import load_dataset
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    args = parser.parse_args()
    return args


def load_pickapic_dataset(cache_dir=None):
    try:
        print("⏳ Loading dataset...")
        dataset = load_dataset(
            "yuvalkirstain/pickapic_v2",
            cache_dir=cache_dir
        )
        print("✅ Dataset loaded successfully!")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


def info(dataset):
    print("\n📊 Dataset Structure:")
    print(f"- Train samples: {len(dataset['train'])}")
    
    print("\n📝 Sample columns:")
    print(dataset['train'].column_names)

    print("\n🔍 Sample data:")
    print(f"caption: {dataset['train']['caption'][0]}")


def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[args.caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{args.caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def preprocess_train(examples):
    all_pixel_values = []
    for col_name in ['jpg_0', 'jpg_1']:
        images = [Image.open(io.BytesIO(im_bytes)).convert("RGB")
            for im_bytes in examples[col_name]]
        pixel_values = [train_transforms(image) for image in images]
        all_pixel_values.append(pixel_values)
    # Double on channel dim, jpg_y then jpg_w
    im_tup_iterator = zip(*all_pixel_values)
    combined_pixel_values = []
    for im_tup, label_0 in zip(im_tup_iterator, examples['label_0']):
        if label_0==0:
            im_tup = im_tup[::-1]
        combined_im = torch.cat(im_tup, dim=0) # no batch dim
        combined_pixel_values.append(combined_im)
    examples["pixel_values"] = combined_pixel_values
    examples["input_ids"] = tokenize_captions(examples)
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return_d =  {"pixel_values": pixel_values}
    return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
                
    return return_d


args = parse_args()

train_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
        transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

args.pretrained_model_name_or_path = '/ifs/root/ipa01/110/user_110003/download/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14'

tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)

if __name__ == "__main__":
    DATA_CACHE_DIR="/ifs/root/data/common/Pick-a-Pic/picapic-v2"
    dataset = load_pickapic_dataset(cache_dir=DATA_CACHE_DIR)
    info(dataset)
    train_dataset = dataset['train'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=16,
        collate_fn=collate_fn,
        drop_last=True
    )
    for step, batch in enumerate(train_dataloader):
        print(batch['pixel_values'].shape)
        print(batch['input_ids'].shape)
        # now you get <x^w x^l c> you can imlement your preference model!
        break
