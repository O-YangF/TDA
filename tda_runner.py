import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *

# my experiment 
# 修改CacheMonitor类
class CacheMonitor:
    def __init__(self, cache_type, max_history=100):
        self.cache_type = cache_type
        self.history = deque(maxlen=max_history)
        self.entropy_stats = {'max': -np.inf, 'min': np.inf}
        self.total_replacements = 0  # 新增累积计数器
        
    def record(self, old_cls, new_cls, old_entropy, new_entropy):
        """记录单次替换事件"""
        # 仅当发生替换时（old_cls存在）才计数
        if old_cls is not None:
            self.total_replacements += 1
            
        record = {
            'old_class': old_cls,
            'new_class': new_cls,
            'old_entropy': old_entropy,
            'new_entropy': new_entropy,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(record)
        self._update_entropy(new_entropy)
        
    def _update_entropy(self, entropy):
        self.entropy_stats['max'] = max(self.entropy_stats['max'], entropy)
        self.entropy_stats['min'] = min(self.entropy_stats['min'], entropy)
        
    def wandb_log(self, step):
        """提交监控数据到wandb"""
        if not self.history:
            return
        
        # 记录极值和累积交换次数
        wandb.log({
            f"{self.cache_type}_cache/max_entropy": self.entropy_stats['max'],
            f"{self.cache_type}_cache/min_entropy": self.entropy_stats['min'],
            f"{self.cache_type}_cache/cumulative_replaces": self.total_replacements  # 新增累积曲线
        }, step=step)


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, monitor=None, include_prob_map=False):
    """更新缓存并记录替换事件"""
    item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
    
    if pred not in cache:
        cache[pred] = []
        
    if len(cache[pred]) < shot_capacity:
        cache[pred].append(item)
        if monitor:
            monitor.record(None, pred, None, features_loss[1].item())
    else:
        # 找到被替换的样本
        old_items = sorted(cache[pred], key=lambda x: x[1])
        replaced_item = old_items[-1]
        
        # 仅当新样本损失更小时替换
        if features_loss[1] < replaced_item[1]:
            old_cls = pred  # 假设缓存按类别组织
            old_entropy = replaced_item[1].item()
            new_entropy = features_loss[1].item()
            
            cache[pred][-1] = item
            cache[pred].sort(key=lambda x: x[1])
            
            if monitor:
                monitor.record(old_cls, pred, old_entropy, new_entropy)

    return cache


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

# 修改run_test_tda函数
def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        pos_monitor = CacheMonitor("positive") if pos_cfg['enabled'] else None
        neg_monitor = CacheMonitor("negative") if neg_cfg['enabled'] else None
        
        # 解包参数
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        # 测试时适应
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

            if pos_enabled:
                pos_cache = update_cache(
                    pos_cache, pred, [image_features, loss], 
                    pos_params['shot_capacity'], pos_monitor
                )

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                neg_cache = update_cache(
                    neg_cache, pred, [image_features, loss, prob_map],
                    neg_params['shot_capacity'], neg_monitor, True
                )

            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
                
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            
            # 记录监控数据
            step = i + 1
            wandb.log({
                "Averaged test accuracy": sum(accuracies)/len(accuracies),
            }, step=step)
            
            if pos_monitor:
                pos_monitor.wandb_log(step)
            if neg_monitor:
                neg_monitor.wandb_log(step)

            if i%1000==0:
                print(f"---- TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----")
                
        final_acc = sum(accuracies)/len(accuracies)
        print(f"---- Final TDA's test accuracy: {final_acc:.2f}. ----")   
        return final_acc



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="TDA-EXPERIMENT0314", config=cfg, group=group_name, name=run_name)

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()