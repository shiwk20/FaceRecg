import argparse
from utils import get_logger
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
from tqdm import tqdm
import torch.distributed as dist
torch.autograd.set_detect_anomaly(True)


def main():
    logger = get_logger('train')
    parse = argparse.ArgumentParser()
    # for distributed training
    parse.add_argument('--local_rank', dest = 'local_rank', type = int, default = -1)
    parse.add_argument('--config', dest = 'config', type = str, default = '')
    args = parse.parse_args()
    print('args.local_rank: ', args.local_rank)
    global device
    device = torch.device("cuda", args.local_rank)
    
    torch.distributed.init_process_group(backend='nccl')
    
    # model
    cur_epoch = 0
    config = OmegaConf.load(args.config)
    model = instantiation(config.model)
    if os.path.isfile(model.ckpt_path):
        cur_epoch, optim_state_dict = model.load_ckpt(model.ckpt_path, logger)
    
    model.train()
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters = False)
    
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
    criterion = instantiation(config.loss)
    
    # optimizer
    params = list(model.module.parameters())
    optimizer = torch.optim.AdamW(params = params, lr = config.optimizer.lr)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = config.optimizer.step_size, gamma = config.optimizer.gamma)
    
    # start loop
    
    best_accuracy = 0
    try:
        for epoch in range(cur_epoch, model.module.max_epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            # train
            model.train()
            epoch_loss = train(optimizer, scheduler, epoch, model, criterion, train_dataloader, logger)

            if dist.get_rank() == 0 and (epoch + 1) % model.module.val_interval == 0:
                # validation
                model.eval()
                accuracy = evaluate(model, val_dataloader, logger, device)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # save ckpt
                    model.module.save_ckpt(model.module.save_ckpt_path, epoch + 1, best_accuracy, optimizer, logger)
                logger.info('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}, Best Accuracy: {:.4f}'.format(epoch, epoch_loss, accuracy, best_accuracy))
            else:
                logger.info('Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))
    except Exception as e:
        print(e)
    finally:
        if dist.get_rank() == 0:
            logger.info('Final Best Accuracy: {:.4f}'.format(best_accuracy))
            model.module.save_ckpt(model.module.save_ckpt_path, -1, best_accuracy, optimizer, logger)
    
def train(optimizer, scheduler, epoch, model, criterion, train_dataloader, logger):
    # loss of an epoch
    train_loss = 0
    counter = 0
    
    tqdm_iter = tqdm(train_dataloader, leave=False)
    for batch in tqdm_iter:
        anc_img, pos_img, neg_img = batch['anc'].to(device), batch['pos'].to(device), batch['neg'].to(device)
        
        anc_fea, pos_fea, neg_fea = model(anc_img), model(pos_img), model(neg_img)
        
        hard_indexes = criterion.get_hard(anc_fea, pos_fea, neg_fea)
        if len(hard_indexes) == 0:
            continue
        counter += len(hard_indexes)
        ans_fea_hard, pos_fea_hard, neg_fea_hard = anc_fea[hard_indexes], pos_fea[hard_indexes], neg_fea[hard_indexes]
        
        loss = criterion(ans_fea_hard, pos_fea_hard, neg_fea_hard)
        train_loss += loss.item() 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tqdm_iter.set_description('Epoch: {}, Loss: {:.4f}, lr: {:.6f}'.format(epoch, loss / len(hard_indexes), optimizer.param_groups[0]['lr']))
    
    scheduler.step()
    return train_loss / counter
    
    
if __name__ == '__main__':
    main()
    