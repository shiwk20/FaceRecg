import argparse
from utils import get_logger, L2_dist, get_train_indexes
from torch import nn
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset.divide_train_val import divide_train_val
from torch.optim import lr_scheduler
from evaluate import evaluate


train_img_path = 'data/train/training_set'

class triplet_loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, anc, pos, neg):
        pos_dist = L2_dist(anc, pos)
        neg_dist = L2_dist(anc, neg)
        return torch.mean(max(pos_dist - neg_dist + self.alpha, 0))
        

def main():
    logger = get_logger()
    parse = argparse.ArgumentParser()
    # for distributed training
    parse.add_argument('--local_rank', dest = 'local_rank', type = int, default = -1)
    parse.add_argument('--config', dest = 'config file path', type = str, default = '')
    args = parse.parse_args()
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend = 'nccl', init_method = 'tcp://localhost:23457', rank = args.local_rank, world_size = torch.cuda.device_count())
    
    # model
    cur_epoch = 0
    config = OmegaConf.load(args.config)
    model = instantiation(config.model)
    if os.path.isfile(model.ckpt_path):
        cur_epoch = model.load_ckpt(model.ckpt_path, logger)
    
    model.train()
    model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters = True)
    
    # data
    divide_train_val(config.data.divide_seed)
    
    train_dataset = instantiation(config.data.train)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = False, # must use False, see https://blog.csdn.net/Caesar6666/article/details/126893353
                                sampler = train_sampler,
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    val_dataset = instantiation(config.data.validation)
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = False, # must use False, see https://blog.csdn.net/Caesar6666/article/details/126893353
                                sampler = val_sampler,
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    # criterion
    criterion = {
    'triplet': triplet_loss(model.alpha),
    'classifier': nn.CrossEntropyLoss()
    }[model.loss_type]
    
    # optimize
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params = params, lr = model.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.1 ** (epoch // 10))
    
    # start loop
    for epoch in range(cur_epoch, model.max_epoch):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # train
        model.train()
        train(optimizer, scheduler, model, criterion, train_dataloader, logger)
        
        # validation
        model.eval()
        evaluate()
        
        # save ckpt
        model.save_ckpt(epoch, logger)
    
def train(optimizer, model):
    loss = 0
    for i, data in enumerate(train_dataloader):
        anc_img, pos_img, neg_img, anc_label, pos_label, neg_label = data
        anc_img, pos_img, neg_img, anc_label, pos_label, neg_label = anc_img.cuda(), pos_img.cuda(), neg_img.cuda(), anc_label.cuda(), pos_label.cuda(), neg_label.cuda()
        
        anc_feature, pos_feature, neg_feature = model(anc_img), model(pos_img), model(neg_img)
        loss = criterion(anc_feature, pos_feature, neg_feature)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            logger.info('epoch: {}, iter: {}, train loss: {}'.format(epoch, i, train_loss / (i + 1)))
    
    