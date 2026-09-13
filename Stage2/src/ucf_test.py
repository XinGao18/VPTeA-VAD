import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import VPTeAVAD, load_compatible_state_dict
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, load_ucf_prompt_pairs, UCF_CLASS_NAMES
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option


def apply_map_gate(pair_probabilities, visual_probabilities, gamma):
    if gamma < 0:
        raise ValueError('mAP gate gamma must be non-negative')
    return pair_probabilities * visual_probabilities[:, None] ** gamma


def test(model, testdataloader, maxlen, positive_prompts, negative_prompts, gt, gtsegments, gtlabels, device, map_gate_gamma=0.5):
    model.to(device)
    model.eval()

    visual_scores = []
    pair_scores = []
    pair_scores_by_video = []

    with torch.no_grad():
        for item in testdataloader:
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1, dtype=torch.long)
            for j in range(len(lengths)):
                lengths[j] = min(length, maxlen)
                length -= int(lengths[j])
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, _, logits1, pair_logits = model(
                visual, padding_mask, positive_prompts, lengths.to(device), negative_prompts
            )
            logits1 = logits1.reshape(-1, logits1.shape[-1])[:len_cur]
            pair_logits = pair_logits.reshape(-1, pair_logits.shape[-1])[:len_cur]

            visual_probabilities = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy()
            pair_probabilities = torch.sigmoid(pair_logits).cpu().numpy()
            visual_scores.append(torch.from_numpy(visual_probabilities))
            pair_scores.append(torch.from_numpy(pair_probabilities).max(dim=-1).values)
            map_scores = apply_map_gate(
                pair_probabilities,
                visual_probabilities,
                map_gate_gamma
            )
            pair_scores_by_video.append(np.repeat(map_scores, 16, axis=0))

    visual_scores = torch.cat(visual_scores).numpy()
    pair_scores = torch.cat(pair_scores).numpy()
    visual_frame_scores = np.repeat(visual_scores, 16)
    pair_frame_scores = np.repeat(pair_scores, 16)
    if len(visual_frame_scores) != len(gt) or len(pair_frame_scores) != len(gt):
        raise ValueError(
            f'Frame score length mismatch: visual={len(visual_frame_scores)}, '
            f'pair={len(pair_frame_scores)}, ground_truth={len(gt)}'
        )

    visual_auc = roc_auc_score(gt, visual_frame_scores)
    visual_ap = average_precision_score(gt, visual_frame_scores)
    pair_auc = roc_auc_score(gt, pair_frame_scores)
    pair_ap = average_precision_score(gt, pair_frame_scores)

    print("Visual AUC: ", visual_auc, " Visual AP: ", visual_ap)
    print("Pair-global AUC: ", pair_auc, " Pair-global AP: ", pair_ap)

    dmap, iou = dmAP(
        pair_scores_by_video, gtsegments, gtlabels,
        excludeNormal=True
    )
    for threshold, value in zip(iou, dmap):
        print('mAP@{0:.1f} ={1:.2f}%'.format(threshold, value))
    averageMAP = float(np.mean(dmap))
    print('average mAP: {:.2f}'.format(averageMAP))

    return visual_auc, pair_ap, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = {class_name: class_name for class_name in UCF_CLASS_NAMES}

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    positive_prompts, negative_prompts = load_ucf_prompt_pairs(args.knowledge_base_path)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = VPTeAVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, device, use_prompt_pairs=True)
    model_param = torch.load(args.model_path, map_location=device)
    load_compatible_state_dict(model, model_param)

    test(
        model, testdataloader, args.visual_length, positive_prompts, negative_prompts,
        gt, gtsegments, gtlabels, device,
        map_gate_gamma=args.map_gate_gamma
    )
