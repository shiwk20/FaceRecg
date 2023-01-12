PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 --master_port=$PORT train.py --config configs/triplet.yaml