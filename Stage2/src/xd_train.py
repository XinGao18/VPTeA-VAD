import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model import VPTeAVAD, load_compatible_state_dict
from utils.dataset import XDDataset
from utils.tools import (
    get_xd_batch_pair_label,
    load_xd_prompt_pairs,
    pair_mil_loss,
    pair_ranking_loss,
    topk_mil_loss,
    XD_CLASS_NAMES
)
from xd_test import evaluate
import xd_option


def train(model, train_loader, test_loader, args, device):
    model.to(device)
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    positive_prompts, negative_prompts = load_xd_prompt_pairs(args.knowledge_base_path)
    map_best = -1
    start_epoch = 0

    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        load_compatible_state_dict(model, checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (KeyError, ValueError, RuntimeError):
            print('optimizer checkpoint is incompatible; using fresh optimizer state')
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        map_best = checkpoint.get('map', checkpoint.get('ap', -1))
        print("checkpoint info:")
        print("epoch:", start_epoch, " mAP:", map_best)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        loss_total_visual = 0
        loss_total_pair = 0
        loss_total_rank = 0
        for index, item in enumerate(train_loader):
            visual_features, labels, feat_lengths = item
            visual_features = visual_features.to(device)
            feat_lengths = feat_lengths.to(device)
            pair_labels = get_xd_batch_pair_label(labels).to(device)

            _, _, logits1, pair_logits = model(
                visual_features, None, positive_prompts, feat_lengths, negative_prompts
            )
            binary_labels = pair_labels.max(dim=1, keepdim=True).values
            loss_visual = topk_mil_loss(logits1, binary_labels, feat_lengths)
            loss_pair = pair_mil_loss(pair_logits, pair_labels, feat_lengths)
            loss_rank = pair_ranking_loss(
                pair_logits,
                pair_labels,
                feat_lengths,
                margin=args.ranking_margin
            )
            loss = (
                loss_visual
                + args.pair_loss_weight * loss_pair
                + args.ranking_loss_weight * loss_rank
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total_visual += loss_visual.item()
            loss_total_pair += loss_pair.item()
            loss_total_rank += loss_rank.item()
            step = index * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                print(
                    'epoch: ', epoch + 1, '| step: ', step,
                    '| visual loss: ', loss_total_visual / (index + 1),
                    '| pair loss: ', loss_total_pair / (index + 1),
                    '| pairwise margin loss: ', loss_total_rank / (index + 1)
                )

        scheduler.step()
        auc, ap, average_map = evaluate(
            model, test_loader, args.visual_length, positive_prompts, negative_prompts,
            gt, gtsegments, gtlabels, device,
            map_gate_gamma=args.map_gate_gamma
        )
        if average_map > map_best:
            map_best = average_map
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'map': map_best,
                'auc': auc,
                'ap': ap,
                'map_gate_gamma': args.map_gate_gamma,
                'cross_attention_alpha': float(model.get_cross_attention_alpha())
            }
            torch.save(checkpoint, args.checkpoint_path)

        torch.save(model.state_dict(), 'model/model_cur_xd.pth')

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    torch.save(checkpoint['model_state_dict'], args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = {class_name: class_name for class_name in XD_CLASS_NAMES}
    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VPTeAVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        device,
        use_prompt_pairs=True
    )
    train(model, train_loader, test_loader, args, device)
