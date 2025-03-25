import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import operator

import clip
from utils import *

# my experiment 
class CacheMonitor:
    def __init__(self, cache_type, dataset_name, max_classes=10, max_history=100):
        self.cache_type = cache_type  # "positive" 或 "negative"
        self.dataset_name = dataset_name
        self.max_classes = max_classes  # 监控的最大类别数量，默认为10
        self.monitored_classes = []    # 存储最先遇到的不同类别
        self.history = deque(maxlen=max_history)  # 历史记录
        self.entropy_stats = defaultdict(lambda: {'max': -np.inf, 'min': np.inf})  # 熵统计
        self.total_replacements = 0    # 累计替换次数
        
    def record(self, old_cls, new_cls, old_entropy, new_entropy):
        # 如果有替换操作，增加计数器
        if old_cls is not None:
            self.total_replacements += 1

        # 动态收集前 max_classes 个不同类别
        if new_cls not in self.monitored_classes and len(self.monitored_classes) < self.max_classes:
            self.monitored_classes.append(new_cls)

        # 只记录属于监控类别的熵值
        if new_cls in self.monitored_classes:
            record = {
                'old_class': old_cls,
                'new_class': new_cls,
                'old_entropy': old_entropy,
                'new_entropy': new_entropy,
                'timestamp': datetime.now().isoformat()
            }
            self.history.append(record)
            self._update_entropy(new_cls, new_entropy)
        
    def _update_entropy(self, cls, entropy):
        # 更新指定类别的最大和最小熵值
        self.entropy_stats[cls]['max'] = max(self.entropy_stats[cls]['max'], entropy)
        self.entropy_stats[cls]['min'] = min(self.entropy_stats[cls]['min'], entropy)
        
    def wandb_log(self, step):
        if not self.history:
            return
        


# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # # 记录每个监控类别的最大和最小熵值
        # for cls in self.monitored_classes:
        #     wandb.log({
        #         f"{self.dataset_name}/{self.cache_type}_cache/class_{cls}/max_entropy": self.entropy_stats[cls]['max'],
        #         f"{self.dataset_name}/{self.cache_type}_cache/class_{cls}/min_entropy": self.entropy_stats[cls]['min'],
        #     }, step=step)

        # # 记录累计替换次数
        # wandb.log({
        #     f"{self.dataset_name}/{self.cache_type}_cache/cumulative_replaces": self.total_replacements
        # }, step=step)
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！



    def get_monitored_classes(self):
        # 返回当前监控的类别列表
        return self.monitored_classes


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


def update_cache(cache, pred, features_loss, shot_capacity, monitor=None, include_prob_map=False, similarity_threshold=0.9):
    """更新缓存并记录替换事件"""
    item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
    
    if pred not in cache:
        cache[pred] = []
        
    if len(cache[pred]) < shot_capacity:
        cache[pred].append(item)
        if monitor:
            monitor.record(None, pred, None, features_loss[1].item())
    else:
        # 计算新样本与队列中所有样本的相似度
        similarities = [F.cosine_similarity(item[0].squeeze(0), existing_item[0].squeeze(0), dim=0).item() 
                        for existing_item in cache[pred]]
        max_similarity = max(similarities)

        if max_similarity < similarity_threshold:
            # 找到熵值最大的样本（将被替换）
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


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, dataset_name, max_classes=20, similarity_threshold=0.9):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        pos_monitor = CacheMonitor("positive", dataset_name, max_classes) if pos_cfg['enabled'] else None
        neg_monitor = CacheMonitor("negative", dataset_name, max_classes) if neg_cfg['enabled'] else None
        
        # 解包参数
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        # 测试时适应
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
        #       特征输入      预测结果   损失值   概率图  预测的类别
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)
        #     标签    损失值的熵
            if pos_enabled:
                pos_cache = update_cache(
                    pos_cache, pred, [image_features, loss], 
                    pos_params['shot_capacity'], pos_monitor, similarity_threshold=similarity_threshold
                )

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                neg_cache = update_cache(
                    neg_cache, pred, [image_features, loss, prob_map],
                    neg_params['shot_capacity'], neg_monitor, True, similarity_threshold=similarity_threshold
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

# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # wandb.log({
            #     "Averaged test accuracy": sum(accuracies)/len(accuracies),
            # }, step=step)
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            
            if pos_monitor:
                pos_monitor.wandb_log(step)
            if neg_monitor:
                neg_monitor.wandb_log(step)

            if i%1000==0:
                print(f"---- TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----")
                
        final_acc = sum(accuracies)/len(accuracies)
        print(f"---- Final TDA's test accuracy: {final_acc:.2f}. ----")   
        return final_acc


def objective(trial, default_cfg, test_loader, clip_model, clip_weights, dataset_name, args):
    cfg = default_cfg.copy()

    # Suggest hyperparameter values
    similarity_threshold = trial.suggest_float('similarity_threshold', 0.5, 1.0)
    pos_shot_capacity = trial.suggest_int('pos_shot_capacity', 1, 10)
    pos_alpha = trial.suggest_float('pos_alpha', 0.1, 2.0)
    pos_beta = trial.suggest_float('pos_beta', 1.0, 10.0)
    neg_shot_capacity = trial.suggest_int('neg_shot_capacity', 1, 10)
    neg_alpha = trial.suggest_float('neg_alpha', 0.01, 1.0)
    neg_beta = trial.suggest_float('neg_beta', 0.1, 2.0)
    entropy_lower = trial.suggest_float('entropy_lower', 0.0, 0.5)
    entropy_upper = trial.suggest_float('entropy_upper', 0.5, 1.0)
    mask_lower = trial.suggest_float('mask_lower', 0.0, 0.1)
    mask_upper = trial.suggest_float('mask_upper', 0.1, 1.0)

    # Update configuration
    cfg['positive']['shot_capacity'] = pos_shot_capacity
    cfg['positive']['alpha'] = pos_alpha
    cfg['positive']['beta'] = pos_beta
    cfg['negative']['shot_capacity'] = neg_shot_capacity
    cfg['negative']['alpha'] = neg_alpha
    cfg['negative']['beta'] = neg_beta
    cfg['negative']['entropy_threshold']['lower'] = entropy_lower
    cfg['negative']['entropy_threshold']['upper'] = entropy_upper
    cfg['negative']['mask_threshold']['lower'] = mask_lower
    cfg['negative']['mask_threshold']['upper'] = mask_upper

    # Run model with updated config
    acc = run_test_tda(
        cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, dataset_name, 
        max_classes=20, similarity_threshold=similarity_threshold
    )

# 如果启用了wandb，记录超参数和准确率
    if args.wandb:
        wandb.log({
            "trial": trial.number,
            "similarity_threshold": similarity_threshold,
            "pos_shot_capacity": pos_shot_capacity,
            "pos_alpha": pos_alpha,
            "pos_beta": pos_beta,
            "neg_shot_capacity": neg_shot_capacity,
            "neg_alpha": neg_alpha,
            "neg_beta": neg_beta,
            "entropy_lower": entropy_lower,
            "entropy_upper": entropy_upper,
            "mask_lower": mask_lower,
            "mask_upper": mask_upper,
            "accuracy": acc
        }, step=trial.number)

    return acc


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
        
        default_cfg = get_config_file(config_path, dataset_name)
        print("\nDefault dataset configurations:")
        print(default_cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="TDA-EXPERIMENT0325-2", config=default_cfg, group=group_name, name=run_name)



            
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, dataset_name, 20)

        # if args.wandb:
        #     wandb.log({f"{dataset_name}": acc})
        #     run.finish()
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！




        # NEW: Create and run Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, default_cfg, test_loader, clip_model, clip_weights, dataset_name, args),
            n_trials=50  # Number of trials, adjustable
        )

        # Log best results
        print(f"Best parameters for {dataset_name}: {study.best_params}")
        print(f"Best accuracy for {dataset_name}: {study.best_value}")

        if args.wandb:
            wandb.log({f"{dataset_name}_best_acc": study.best_value})
            run.finish()






if __name__ == "__main__":
    main()