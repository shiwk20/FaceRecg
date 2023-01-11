from sklearn.model_selection import KFold
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch
import numpy as np
from utils import L2_dist
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm



def get_accuracy(threshold, dists, labels):
    pred = []
    for dist in dists:
        pred.append(1) if dist < threshold else pred.append(0)
    pred, labels = np.array(pred), np.array(labels)
    accuracy = np.mean(pred == labels)
    return accuracy

def evaluate(model, val_dataloader, logger, device):
    labels = []
    dists = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='get distance', leave=False):
            A_img, B_img = batch['A'].to(device), batch['B'].to(device)
            A_fea, B_fea = model(A_img), model(B_img)
            distance = L2_dist(A_fea, B_fea)
            
            dists += distance.cpu().numpy().tolist()
            labels += batch['label'].cpu().numpy().tolist()

    Accuracies = []
    Thresholds = []
    kf = KFold(n_splits=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.01)
    
    tqdm_kf = tqdm(kf.split(dists) ,leave=False)
    for train_indexes, test_indexes in tqdm_kf:
        best_accuracy = 0
        for threshold in thresholds:
            accuracy = get_accuracy(threshold, np.array(dists)[train_indexes], np.array(labels)[train_indexes])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        tqdm_kf.set_description('Accuracy: {:.4f} Threshold: {:.4f}'.format(best_accuracy, best_threshold))
        
        Accuracies.append(get_accuracy(best_threshold, np.array(dists)[test_indexes], np.array(labels)[test_indexes]))
        Thresholds.append(best_threshold)

    logger.info('Accuracy: {:.4f} Threshold: {:.4f}'.format(np.mean(accuracy), np.mean(thresholds)))

    return np.mean(accuracy)

if __name__ == '__main__':
    logger = get_logger('evaluate')
    device = 'cpu'
    config = OmegaConf.load('configs/triplet.yaml')
    model = instantiation(config.model)
    if os.path.isfile(model.ckpt_path):
        cur_epoch, optim_state_dict = model.load_ckpt(model.ckpt_path, logger)
    model.eval()
    model.to(device)
    
    val_dataset = instantiation(config.data.validation)
    val_dataloader = DataLoader(val_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = True, # must use False, see https://blog.csdn.net/Caesar6666/article/details/126893353
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    accuracy = evaluate(model, val_dataloader, logger, device)
    logger.info('Accuracy: {:.4f}'.format(accuracy))
    