#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py --config configs \
                                            --wandb-log \
                                            --datasets I \
                                            --backbone ViT-B/16
