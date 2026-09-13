import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import VPTeAVAD, load_compatible_state_dict
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import (
    build_class_balanced_sampler,
    get_batch_pair_label,
    load_ucf_prompt_pairs,
    pair_mil_loss,
    pair_ranking_loss,
    topk_mil_loss,
    UCF_CLASS_NAMES
)
import ucf_option

def train(model, normal_loader, anomaly_loader, testloader, args, device):
    model.to(device)
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    positive_prompts, negative_prompts = load_ucf_prompt_pairs(args.knowledge_base_path)
    map_best = -1
    start_epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        load_compatible_state_dict(model, checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (KeyError, ValueError, RuntimeError):
            print('optimizer checkpoint is incompatible; using fresh optimizer state')
        start_epoch = checkpoint['epoch'] + 1
        map_best = checkpoint.get('map', checkpoint.get('ap', -1))
        print("checkpoint info:")
        print("epoch:", start_epoch, " mAP:", map_best)

    for e in range(start_epoch, args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total3 = 0
        loss_total_rank = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        iterations = min(len(normal_loader), len(anomaly_loader))
        for i in range(iterations):
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = get_batch_pair_label(list(normal_label) + list(anomaly_label)).to(device)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)

            _, _, logits1, pair_logits = model(
                visual_features, None, positive_prompts, feat_lengths, negative_prompts
            )
            binary_labels = text_labels.max(dim=1, keepdim=True).values
            loss1 = topk_mil_loss(logits1, binary_labels, feat_lengths)
            loss3 = pair_mil_loss(pair_logits, text_labels, feat_lengths)
            loss_rank = pair_ranking_loss(
                pair_logits,
                text_labels,
                feat_lengths,
                margin=args.ranking_margin
            )
            loss = (
                loss1
                + args.pair_loss_weight * loss3
                + args.ranking_loss_weight * loss_rank
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total1 += loss1.item()
            loss_total3 += loss3.item()
            loss_total_rank += loss_rank.item()
            step = i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step,
                      '| visual loss: ', loss_total1 / (i+1),
                      '| pair loss: ', loss_total3 / (i+1),
                      '| pairwise margin loss: ', loss_total_rank / (i+1))

        scheduler.step()
        AUC, AP, averageMAP = test(
            model, testloader, args.visual_length, positive_prompts, negative_prompts,
            gt, gtsegments, gtlabels, device,
            map_gate_gamma=args.map_gate_gamma
        )
        if averageMAP > map_best:
            map_best = averageMAP
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': map_best,
                'auc': AUC,
                'ap': AP,
                'map_gate_gamma': args.map_gate_gamma,
                'cross_attention_alpha': float(model.get_cross_attention_alpha())
            }
            torch.save(checkpoint, args.checkpoint_path)

        torch.save(model.state_dict(), 'model/model_cur.pth')

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = {class_name: class_name for class_name in UCF_CLASS_NAMES}

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_sampler, _, anomaly_counts = build_class_balanced_sampler(
        anomaly_dataset.df['label'].tolist(),
        power=args.anomaly_sampling_power
    )
    print('anomaly class counts:', dict(anomaly_counts))
    print('anomaly sampling power:', args.anomaly_sampling_power)
    anomaly_loader = DataLoader(
        anomaly_dataset,
        batch_size=args.batch_size,
        sampler=anomaly_sampler,
        drop_last=True
    )

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VPTeAVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, device, use_prompt_pairs=True)

    train(model, normal_loader, anomaly_loader, test_loader, args, device)
