import datetime
import argparse
from utils import get_logger
from torch import nn
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation
from torch.utils.data import DataLoader
from utils import divide_train_val
from torch.optim import lr_scheduler
from evaluate import evaluate
from tqdm import tqdm

    
def update_dataloader(divide_seed, align_type, config, local_rank = 0):
    train_indexes, val_indexes = divide_train_val(seed = divide_seed, align_type = align_type,local_rank = local_rank)
    train_indexes = {'train_indexes': train_indexes}
    val_indexes = {'val_indexes': val_indexes}
    
    train_dataset = instantiation(config.data.train, train_indexes)
    train_dataloader = DataLoader(train_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = True, 
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    val_dataset = instantiation(config.data.validation, val_indexes)
    val_dataloader = DataLoader(val_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = True, 
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    return train_dataloader, val_dataloader

def main():
    logger = get_logger('train')
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', help = 'align type(mtcnn or landmark), default: mtcnn', type = str, default = 'mtcnn')
    args = parser.parse_args()
    config_path = f'configs/train_{args.type}.yaml'
    
    global device
    device = 'cuda:7'
    
    # model
    cur_epoch = 0
    config = OmegaConf.load(config_path)
    model = instantiation(config.model)
    optim_state_dict = None
    if os.path.isfile(model.ckpt_path):
        cur_epoch, optim_state_dict = model.load_ckpt(model.ckpt_path, logger)
    model.train()
    model.to(device)

    # criterion
    criterion = instantiation(config.loss)
    
    # optimizer
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params = params, lr = config.optimizer.max_lr)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
    lambda_lr = lambda epoch: ((config.optimizer.base_lr - config.optimizer.max_lr) / config.optimizer.warm_up_epochs * epoch + config.optimizer.max_lr) / config.optimizer.max_lr if epoch < config.optimizer.warm_up_epochs else (config.optimizer.min_lr + 0.5 * (config.optimizer.base_lr - config.optimizer.min_lr) * (1 + np.cos((epoch - config.optimizer.warm_up_epochs) / (config.optimizer.max_epochs - config.optimizer.warm_up_epochs) * np.pi))) / config.optimizer.max_lr
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_lr, last_epoch=cur_epoch - 1)
    # start loop
    best_accuracy = 0
    counter_flag = 0
    accuracy_flag = 0
    try:
        for epoch in range(cur_epoch, config.optimizer.max_epochs):
            train_dataloader, val_dataloader = update_dataloader(divide_seed = epoch, align_type=args.type,config = config)
            # train
            model.train()
            epoch_loss, Flag = train(optimizer, scheduler, epoch, model, criterion, train_dataloader)
            if (epoch + 1) % config.val_interval == 0:
                # validation
                model.eval()
                accuracy, _ = evaluate(model, val_dataloader, logger, device)
                if accuracy > best_accuracy:
                    accuracy_flag = 0
                    best_accuracy = accuracy
                    # save ckpt
                    model.save_ckpt(model.save_ckpt_path, epoch + 1, best_accuracy, optimizer, logger, start_time)
                else:
                    accuracy_flag += 1
                    print('accuracy_flag: {}'.format(accuracy_flag))
                    if accuracy_flag >= 5:
                        break
                logger.info('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}, Best Accuracy: {:.4f}'.format(epoch, epoch_loss, accuracy, best_accuracy))
            else:
                logger.info('Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))

            if Flag:
                counter_flag += 1
                print('counter_flag: {}'.format(counter_flag))
            else:
                counter_flag = 0
            if counter_flag >= 5:
                break
    except Exception as e:
        print(e)
    finally:
        logger.info('Final Best Accuracy: {:.4f}'.format(best_accuracy))
        model.save_ckpt(model.save_ckpt_path, -1, best_accuracy, optimizer, logger, start_time)
    
def train(optimizer, scheduler, epoch, model, criterion, train_dataloader):
    # loss of an epoch
    ave_loss = 0
    counter = 0
    Flag = False
    tqdm_iter = tqdm(train_dataloader, leave=False)
    for batch in tqdm_iter:
        anc_img, pos_img, neg_img = batch['anc'].to(device), batch['pos'].to(device), batch['neg'].to(device)
        anc_fea, pos_fea, neg_fea = model(anc_img), model(pos_img), model(neg_img)
        
        loss, count = criterion(anc_fea, pos_fea, neg_fea)
        
        ave_loss += loss.item() * count
        counter += count
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm_iter.set_description('Epoch: {}, Loss: {:.4f}, lr: {:.6f}, Count: {}'.format(epoch, loss, optimizer.param_groups[0]['lr'], count))
    
    scheduler.step()
    print('counter: ', counter)
    if counter < 50:
        Flag = True
    return (ave_loss / counter, Flag) if counter != 0 else (0, Flag)
    
    
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m-%d_%H-%M-%S")
    
    main()
    