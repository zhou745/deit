python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dist-eval --model deit_tiny_patch16_224 --batch-size 256\
 --data-path ~/datasets/imagenet_val --output_dir ./training/baseline_tiny_v1_val --epochs 300 --lr 5e-4
