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
    
    """
    更新缓存，结合熵值和相似度进行更精细的替换。
    Args:
        cache: 当前缓存 (dict: class_id -> list of [feature, loss, Optional[prob_map]])
        pred: 当前样本的预测类别
        features_loss: 包含 [图像特征, 损失/熵, Optional[概率图]] 的列表
        shot_capacity: 每个类别缓存的最大容量
        monitor: CacheMonitor实例，用于记录
        include_prob_map: 是否在缓存项中包含概率图 (用于负缓存)
        similarity_threshold: 用于判断样本是否高度相似的余弦相似度阈值
    """
        
    # 准备要缓存的项
    item_to_cache = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
    new_feature = item_to_cache[0]
    new_loss = item_to_cache[1] # 使用 loss/entropy 作为判断标准
    
    if pred not in cache:
        cache[pred] = []
        
    if len(cache[pred]) < shot_capacity:
        cache[pred].append(item_to_cache)
        if monitor:
            monitor.record(None, pred, None, features_loss[1].item())
    else:
        
        higher_loss_group = [] # 存储 (index, sample)
        lower_loss_group = []  # 存储 (index, sample)
    
        # 1. 分离缓存为两组
        for idx, existing_item in enumerate(cache[pred]):
            if existing_item[1] > new_loss:
                higher_loss_group.append((idx, existing_item))
            else:
                lower_loss_group.append((idx, existing_item))
        
        # 2. 检查 higher_loss_group 中是否存在高相似度样本
        highly_similar_in_higher = [] # 存储 (index, sample, similarity)
        for idx, existing_item in higher_loss_group:
            similarity = F.cosine_similarity(new_feature.squeeze(0), existing_item[0].squeeze(0), dim=0).item()
            if similarity > similarity_threshold:
                highly_similar_in_higher.append((idx, existing_item, similarity))

        # Action A: 如果找到高相似度样本，替换其中熵最高的那个
        if highly_similar_in_higher:
            # 找到这些高相似度样本中熵最高的那个
            best_candidate_tuple = max(highly_similar_in_higher, key=lambda x: x[1][1])

            target_idx = best_candidate_tuple[0]
            item_to_replace = best_candidate_tuple[1]
            
            old_entropy = item_to_replace[1].item()
            new_entropy = new_loss.item()
            
            # 执行替换
            cache[pred][target_idx] = item_to_cache
            
            if monitor:
                monitor.record(pred, pred, old_entropy, new_entropy)
            # 替换发生，结束此样本的处理
            return cache 
        
        # 3. 如果 Action A 未执行，检查 lower_entropy_group 中是否存在高相似度样本
        found_high_similarity_in_lower = False
        for idx, existing_item in lower_loss_group:
            similarity = F.cosine_similarity(new_feature.squeeze(0), existing_item[0].squeeze(0), dim=0).item()
            if similarity > similarity_threshold:
                found_high_similarity_in_lower = True
                break # 找到一个就足够判断


        # Action B: 如果找到，直接返回，不做任何更改
        if found_high_similarity_in_lower:
            # 不做任何操作，因为新样本与一个已经很好的样本太相似
            return cache

        # 4. 如果 Action A 和 B 都未执行 (即新样本与现有样本都不高度相似)
        # Action C: 检查 higher_entropy_group 是否为空。如果不为空，替换其中熵最高的那个样本
        if higher_loss_group: # 确保有比新样本更差的样本存在
             # 找到 higher_entropy_group 中熵最高的样本
            target_idx, item_to_replace = max(higher_loss_group, key=lambda x: x[1][1]) # x[1]是sample, x[1][1]是loss

            old_entropy = item_to_replace[1].item()
            new_entropy = new_loss.item()

            # 执行替换
            cache[pred][target_idx] = item_to_cache

            if monitor:
                monitor.record(pred, pred, old_entropy, new_entropy)
        # else: # 如果 higher_entropy_group 为空，意味着新样本不比缓存中任何样本更好，则不执行任何操作

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


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, dataset_name, max_classes=20, pos_similarity_threshold=0.9, neg_similarity_threshold=0.8):
    """Run test-time adaptation on the given dataset."""
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
                    pos_params['shot_capacity'], pos_monitor, similarity_threshold=pos_similarity_threshold
                )

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                neg_cache = update_cache(
                    neg_cache, pred, [image_features, loss, prob_map],
                    neg_params['shot_capacity'], neg_monitor, True, similarity_threshold=neg_similarity_threshold
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
            #     f"{num_step}/Averaged test accuracy": sum(accuracies)/len(accuracies),
            # }, step=step)
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            
            if pos_monitor:
                pos_monitor.wandb_log(step)
            if neg_monitor:
                neg_monitor.wandb_log(step)

            if i%1000==0:
                print(f"---- TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----")

            # if i == 30000:
            #     break
                
        final_acc = sum(accuracies)/len(accuracies)
        print(f"---- Final TDA's test accuracy: {final_acc:.2f}. ----")   
        return final_acc


def objective(trial, default_cfg, test_loader, clip_model, clip_weights, dataset_name, args):
    cfg = default_cfg.copy()

    # Suggest hyperparameter values
    pos_similarity_threshold = trial.suggest_float('pos_similarity_threshold', 0.7, 1.0)  # 正缓存阈值
    neg_similarity_threshold = trial.suggest_float('neg_similarity_threshold', 0.7, 1.0)  # 负缓存阈值
    pos_shot_capacity = trial.suggest_int('pos_shot_capacity', 2, 10)
    # pos_alpha = trial.suggest_float('pos_alpha', 0.1, 2.0)
    # pos_beta = trial.suggest_float('pos_beta', 1.0, 10.0)
    neg_shot_capacity = trial.suggest_int('neg_shot_capacity', 2, 10)
    # neg_alpha = trial.suggest_float('neg_alpha', 0.01, 1.0)
    # neg_beta = trial.suggest_float('neg_beta', 0.1, 2.0)
    # entropy_lower = trial.suggest_float('entropy_lower', 0.0, 0.5)
    # entropy_upper = trial.suggest_float('entropy_upper', 0.5, 1.0)
    # mask_lower = trial.suggest_float('mask_lower', 0.0, 0.1)
    # mask_upper = trial.suggest_float('mask_upper', 0.1, 1.0)

    # Update configuration
    cfg['positive']['shot_capacity'] = pos_shot_capacity
    # cfg['positive']['alpha'] = pos_alpha
    # cfg['positive']['beta'] = pos_beta
    cfg['negative']['shot_capacity'] = neg_shot_capacity
    # cfg['negative']['alpha'] = neg_alpha
    # cfg['negative']['beta'] = neg_beta
    # cfg['negative']['entropy_threshold']['lower'] = entropy_lower
    # cfg['negative']['entropy_threshold']['upper'] = entropy_upper
    # cfg['negative']['mask_threshold']['lower'] = mask_lower
    # cfg['negative']['mask_threshold']['upper'] = mask_upper

    # Run model with updated config
    acc = run_test_tda(
        cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, dataset_name, 
        max_classes=20,pos_similarity_threshold=pos_similarity_threshold, neg_similarity_threshold=neg_similarity_threshold
    )

# 如果启用了wandb，记录超参数和准确率
    if args.wandb:
        wandb.log({
            "trial": trial.number,
            "pos_similarity_threshold": pos_similarity_threshold,
            "neg_similarity_threshold": neg_similarity_threshold,
            "pos_shot_capacity": pos_shot_capacity,
            # "pos_alpha": pos_alpha,
            # "pos_beta": pos_beta,
            "neg_shot_capacity": neg_shot_capacity,
            # "neg_alpha": neg_alpha,
            # "neg_beta": neg_beta,
            # "entropy_lower": entropy_lower,
            # "entropy_upper": entropy_upper,
            # "mask_lower": mask_lower,
            # "mask_upper": mask_upper,
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
            run = wandb.init(project="TDA-EXPERIMENT0403-I", config=default_cfg, group=group_name, name=run_name)



            
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # acc = run_test_tda(default_cfg['positive'], default_cfg['negative'], test_loader, clip_model, clip_weights, dataset_name, 20)

        # if args.wandb:
        #     wandb.log({f"{dataset_name}": acc})
        #     run.finish()
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！



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
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！




if __name__ == "__main__":
    main()