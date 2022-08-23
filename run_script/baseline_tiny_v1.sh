python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dist-eval --model deit_tiny_patch16_224 --batch-size 256\
 --data-path ~/datasets/imagenet --output_dir ./training/baseline_tiny_v1 --epochs 300 --lr 5e-4
