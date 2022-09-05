python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dist-eval --model deit_t8_patch16_224 --batch-size 256 --num_workers 10\
 --data-path ~/datasets/imagenet --output_dir ./training/baseline_t8_v1_noaug --epochs 300 --lr 5e-4
