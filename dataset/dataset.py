from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
from PIL import Image

def process_image(image, rsize):
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)   
    return image
    
# produce triplet images
class TripletTrainDataset(Dataset):
    def __init__(self, idx_path, triplets_num, align_type, align_size, rsize) -> None:
        super().__init__()
        self.img_path = f'data/train/{align_type}/align{align_size}x{align_size}'
        self.idx_path = idx_path
        self.triplets_num = triplets_num
        self.rsize = rsize
        self.train_indexes = json.load(open(idx_path, 'r'))
        self.triplets = {}
        
        for i in range(triplets_num):
            print('triplet: ', i)
            tmp_triplet = self.get_triplet()
            self.triplets[i] = tmp_triplet
    
    def get_triplet(self):
        triplet = {}
        anc_name = random.choice(list(self.train_indexes.keys()))
        while len(self.train_indexes[anc_name]) < 2:
            anc_name = random.choice(list(self.train_indexes.keys()))
        anc_idx, pos_idx = random.sample(self.train_indexes[anc_name], 2) 
        
        neg_name = random.choice(list(self.train_indexes.keys()))
        while neg_name == anc_name:
            neg_name = random.choice(list(self.train_indexes.keys()))
        neg_idx = random.choice(self.train_indexes[neg_name])
        
        triplet['anc_path'] = os.path.join(anc_name, anc_idx)
        triplet['pos_path'] = os.path.join(anc_name, pos_idx)
        triplet['neg_path'] = os.path.join(neg_name, neg_idx)
        return triplet
        
    def __len__(self):
        return self.triplets_num
            
    def __getitem__(self, index):
        batch = {}
        
        anc_image = Image.open(self.triplets[index]['anc_path'])
        anc_image = process_image(anc_image)
        pos_image = Image.open(self.triplets[index]['pos_path'])
        pos_image = process_image(pos_image)
        neg_image = Image.open(self.triplets[index]['neg_path'])
        neg_image = process_image(neg_image)
        
        batch['anc'], batch['pos'], batch['neg'] = anc_image, pos_image, neg_image
        return batch
    

# produce pair images
class TripletValDataset(Dataset):
    def __init__(self, idx_path, pairs_num, align_type, align_size, rsize) -> None:
        super().__init__()
        self.img_path = f'data/train/{align_type}/align{align_size}x{align_size}'
        self.idx_path = idx_path
        self.pairs_num = pairs_num
        self.rsize = rsize
        self.val_indexes = json.load(open(idx_path, 'r'))
        self.pairs = {}
        
        for i in range(pairs_num // 2):
            tmp_pair = self.get_pos_pair()
            # remove duplication
            while tmp_pair in self.pairs.values():
                tmp_pair = self.get_pos_pair()
            self.pairs[i] = tmp_pair
            
        for i in range(pairs_num // 2):
            tmp_pair = self.get_neg_pair()
            # remove duplication
            while tmp_pair in self.pairs.values():
                tmp_pair = self.get_neg_pair()
            self.pairs[i + pairs_num // 2] = tmp_pair
    
    def get_pos_pair(self):
        pair = {}
        anc_name = random.choice(list(self.val_indexes.keys()))
        while len(self.val_indexes[anc_name]) < 2:
            anc_name = random.choice(list(self.val_indexes.keys()))
        anc_idx, pos_idx = random.sample(self.val_indexes[anc_name], 2)

        pair['A'] = os.path.join(anc_name, anc_idx)
        pair['B'] = os.path.join(anc_name, pos_idx)
        pair['label'] = 1
        return pair
        
    def get_neg_pair(self):
        pair = {}
        anc_name, neg_name = random.sample(list(self.val_indexes.keys()), 2)
        anc_idx = random.choice(self.val_indexes[anc_name])
        neg_idx = random.choice(self.val_indexes[neg_name])
        
        pair['A'] = os.path.join(anc_name, anc_idx)
        pair['B'] = os.path.join(neg_name, neg_idx)
        pair['label'] = 0
        return pair
        
        
    def __getitem__(self, index):
        batch = {}
        
        A_image = Image.open(self.pairs[index]['A'])
        A_image = process_image(A_image)
        B_image = Image.open(self.pairs[index]['B'])
        B_image = process_image(B_image)
        
        batch['A'], batch['B'], batch['label'] = A_image, B_image, self.pairs[index]['label']
        return batch