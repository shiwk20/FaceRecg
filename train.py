
import builtins
import datetime
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



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
def main():
    logger = get_logger('train')
    parser = argparse.ArgumentParser()
    # for distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--config', dest = 'config', type = str, default = '')
    args = parser.parse_args()
    init_distributed_mode(args)
    setup_for_distributed(is_main_process())
    global device
    device = get_rank()
    
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
    