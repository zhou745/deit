python -m torch.distributed.launch --nproc_per_node=8 --use_env main_score.py --dist-eval --model deit_tiny_patch16_224 --batch-size 256 --num_workers 10 --layer_decay 0.2\
 --data-path ~/datasets/imagenet --output_dir ./training/score_tune_tiny_v1_noaug_lr_4e-5_decay_0.2 --epochs 100 --lr 4e-5 --min-lr 1e-6 --tune_model ./training/baseline_tiny_v1_noaug/checkpoint.pth
