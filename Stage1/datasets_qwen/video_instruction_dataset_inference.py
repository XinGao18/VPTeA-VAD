import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import decord
from decord import VideoReader, cpu
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
import math
from torchvision import transforms
import pdb
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel
import pytorch_lightning as pl
import itertools

from knowledge_base import extract_ucf_category, extract_xd_categories


def build_transform(input_size, mean, std):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def dynamic_preprocess(frames, image_size=448, grid_size=1):
    """
    Takes grid_size*grid_size frames and stitches them into a single image of size `image_size`.
    
    Args:
    - frames: List of frames to stitch into a grid.
    - image_size: Final size of the stitched image (default 448x448).
    - grid_size: Number of frames per row and column (default 2x2).

    Returns:
    - stitched_image: A stitched image of size `image_size` with the frames arranged in a grid.
    """
    tile_size = image_size // grid_size
    num_frames = grid_size * grid_size

    assert len(frames) == num_frames, f"{num_frames} frames are required to stitch into a {grid_size}x{grid_size} grid."

    # Resize each frame to fit within the grid
    resized_frames = [frame.resize((tile_size, tile_size)) for frame in frames]
    
    # Create a blank canvas for the final stitched image
    stitched_image = Image.new('RGB', (image_size, image_size))

    # Paste each frame into the grid
    for idx, frame in enumerate(resized_frames):
        # Calculate row and column for each frame based on grid_size
        row = idx // grid_size
        col = idx % grid_size
        stitched_image.paste(frame, (col * tile_size, row * tile_size))

    return stitched_image

class Video_Instruct_Dataset_Inference(Dataset):
    def __init__(self, vis_root, ann_root, data_type='video',sampling_rate=16, snippet_len=10):
        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        
        self.vis_root = vis_root
        self.resize_size = 448
        self.sampling_rate = sampling_rate
        self.snippet_len = snippet_len
        self.data_type = data_type
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        self.transform = build_transform(self.resize_size, self.IMAGENET_MEAN, self.IMAGENET_STD)
        self.cls_labels = [self._extract_cls_labels(sample) for sample in self.annotation]

    def _extract_cls_labels(self, sample):
        if 'video_name' in sample or 'path' in sample:
            return extract_xd_categories(sample)
        video_ref = sample.get('video', '')
        if '_label_' in os.fspath(video_ref).casefold():
            return extract_xd_categories(sample)
        category = extract_ucf_category(sample)
        return {category} if category is not None else set()

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        num_retries = 10  
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_ref = sample.get('video', sample.get('video_name', sample.get('path')))
                video_name = os.path.splitext(os.path.basename(os.fspath(video_ref)))[0]
                video_path = os.path.join(self.vis_root, video_name)
                video_label_vad = 0 if not self.cls_labels[index] else 1
                    
                vlen = sample['length']

            except:
                print(f"Failed to load examples with video: {video_path}. Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return video_path, video_name, vlen
