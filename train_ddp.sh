CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 --master_port=23756 train_ddp.py --type mtcnn
