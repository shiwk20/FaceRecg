from sklearn.model_selection import KFold
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch
import numpy as np
from utils import L2_dist



def get_accuracy(threshold, dists, labels):
    pred = []
    for dist in dists:
        pred.append(1) if dist < threshold else pred.append(0)
    pred, labels = np.array(pred), np.array(labels)
    accuracy = np.mean(pred == labels)
    return accuracy

def evaluate(model, val_dataloader, logger):
    labels = []
    dists = []

    with torch.no_grad():
        for batch in val_dataloader:
            A_img, B_img = batch['A'].cuda(), batch['B'].cuda()
            A_fea, B_fea = model(A_img), model(B_img)
            distance = L2_dist(A_fea, B_fea)
            
            dists += distance.cpu().numpy().tolist()
            labels += batch['label'].cpu().numpy().tolist()

    accuracies = []
    thresholds = []
    kf = KFold(n_splits=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.01)
    
    for train_indexes, test_indexes in kf.split(dists):
        best_accuracy = 0
        for threshold in thresholds:
            accuracy = get_accuracy(threshold, dists[train_indexes], labels[train_indexes])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        accuracies.append(get_accuracy(best_threshold, dists[test_indexes], labels[test_indexes]))
        thresholds.append(best_threshold)

    logger.info('Accuracy: {:.4f} Threshold: {:.4f}'.format(np.mean(accuracy), np.mean(thresholds)))

    return np.mean(accuracy)