python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dist-eval --output_dir ./trainings/baseline_tiny_ls\
       --seed 0 --opt fusedlamb --smoothing 0.0 --reprob 0.0 --repeated-aug --bce-loss --ThreeAugment
