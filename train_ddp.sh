CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port=23755 train_ddp.py --type mtcnn
