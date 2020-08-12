import numpy as np
import torch
import scipy.io
import os
import ipdb
import pickle
import h5py
#from utils import LLE_utils
#from utils import KNN_utils
from torch.utils.data import Dataset, DataLoader

class Dataset_setup(Dataset):
    def __init__(self,data, attrs, labels):
        self.data = data
        self.attrs = attrs
        self.labels = labels
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        sample_idx = self.data[idx,:]
        attr_idx = self.labels[idx].astype('int16') -1
        attr = self.attrs[attr_idx,:]
        sample = {'feature': sample_idx, 'attr': attr, 'label': attr_idx}
        
        return sample 

class Dataset_setup2(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        sample_idx = self.data[idx,:]
        labels_idx = self.labels[idx]
        sample = {'feature': sample_idx, 'label': labels_idx}
        return sample
        
class Dataset_setup_batch(Dataset):
    def __init__(self, data, attrs, labels):
        self.data = data
        self.attrs = attrs
        self.labels = labels
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_idx = self.data[idx]
        attr_idx = self.labels[idx].astype('int16') -1
        attr_ = self.attrs[attr_idx[0]]
        attr = np.tile(attr_, (sample_idx.shape[0],1))
        sample = {'feature': sample_idx, 'attr': attr, 'label': attr_idx}
        
        return sample 
        

class Imagenet(object):
    def __init__(self,
                dataset_name,
                data_path,
                ifnorm = True,
                iftrain = False,
                n_neighbors = 20):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.ifnorm = ifnorm
        self.iftrain = iftrain
        self.n_neighbors = n_neighbors
        self.prepare_data()
        
    def norm_data(self):
        for i in range(self.attrs.shape[0]):
            print('{} {}'.format(i,np.linalg.norm(self.attrs[i,:])))
            self.attrs[i,:] = self.attrs[i,:]/np.linalg.norm(self.attrs[i,:])
        print('norm attributes done!')
    
        for i in range(self.features.shape[0]):
            self.features[i,:] = self.features[i,:]/np.linalg.norm(self.features[i,:])
        
        for i in range(self.features_val.shape[0]):
            self.features_val[i,:] = self.features_val[i,:]/np.linalg.norm(self.features_val[i,:])    
         
        print('norm features done!')
        
        
    
    def prepare_data(self):
        feature_path = os.path.join(self.data_path, "ILSVRC2012_res101_feature.mat")
        attr_path = os.path.join(self.data_path, "ImageNet_w2v.mat")        
        with h5py.File(attr_path, 'r') as f:
            attr_keys = list(f.keys())
            '''
            no_w2v_loc = f['no_w2v_loc']
            wnids = f['wnids']
            words = f['words']
            '''
            w2v = f['w2v']
            self.attrs = w2v[:].T
            

        with h5py.File(feature_path, 'r') as f: 
            dataset_keys = list(f.keys())
            self.features = f['features'][:].T
            self.features_val = f['features_val'][:]
            self.labels = f['labels'][:].T
            self.labels_val = f['labels_val'][:].T     
            #self.visual_features = features 
            #self.visual_labels = labels 
                  
        '''            
        if self.ifnorm:
            self.norm_data()
        '''
        
        train_idx = np.where(self.labels <= 200)[0]
        test_seen_idx = np.where(self.labels_val <=200)[0]
        test_unseen_idx = np.where(self.labels_val>900)[0]
        
        self.train_set = self.features[train_idx, :]
        self.train_labels = self.labels[train_idx, :]
        
        self.test_seen_set = self.features_val[test_seen_idx, :]
        self.test_seen_labels = self.labels_val[test_seen_idx, :]
        
        self.test_unseen_set = self.features_val[test_unseen_idx, :]
        self.test_unseen_labels = self.labels_val[test_unseen_idx, :]
        
        self.val_set = self.test_seen_set
        self.val_labels = self.test_seen_labels
        
        self.seen_labels = np.array(list(range(1,200)))
        self.unseen_labels = np.array(list(range(901, 1000)))
        


        
class AwA2(object):
    def __init__(self,
                 dataset_name,
                 data_path,
                 ifnorm = True,
                 iftrain = False,
                 n_neighbors = 5):
                 
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.ifnorm = ifnorm
        self.iftrain = iftrain
        self.n_neighbors = n_neighbors
        self.prepare_data()
    
    def norm_data(self):
        for i in range(self.visual_features.shape[0]):
            self.visual_features[i,:] = self.visual_features[i,:]/np.linalg.norm(self.visual_features[i,:]) * 1.0
        print('norm features done!')
        
    def prepare_data(self):
        
        feature_path = os.path.join(self.data_path, "res101.mat")
        attr_path = os.path.join(self.data_path, "att_splits.mat")
        
        features = scipy.io.loadmat(feature_path)
        attr = scipy.io.loadmat(attr_path)
        self.visual_features = features['features'].T
        self.visual_labels = features['labels']
        self.attrs = attr['att'].T * 1.0
        self.train_loc = attr['train_loc']
        self.val_loc = attr['val_loc'] 
        self.trainval_loc = attr['trainval_loc']
        self.test_seen_loc = attr['test_seen_loc']
        self.test_unseen_loc = attr['test_unseen_loc']
        
        
        if self.ifnorm:
            self.norm_data()
        
        self.train_set_ = self.visual_features[self.train_loc.reshape(-1)-1,:]
        self.train_labels_ = self.visual_labels[self.train_loc.reshape(-1)-1,:]
        
        self.val_set = self.visual_features[self.val_loc.reshape(-1)-1,:]
        self.val_labels = self.visual_labels[self.val_loc.reshape(-1)-1,:]
        
        self.trainval_set =  self.visual_features[self.trainval_loc.reshape(-1)-1,:]
        self.trainval_labels = self.visual_labels[self.trainval_loc.reshape(-1)-1,:]
               
        self.test_seen_set = self.visual_features[self.test_seen_loc.reshape(-1)-1,:]
        self.test_seen_labels = self.visual_labels[self.test_seen_loc.reshape(-1)-1,:]        
        self.test_seen_labels2 = np.ones(self.test_seen_labels.shape[0]).reshape(-1,1).astype('int16')
        
        self.test_unseen_set = self.visual_features[self.test_unseen_loc.reshape(-1)-1,:]
        self.test_unseen_labels = self.visual_labels[self.test_unseen_loc.reshape(-1)-1,:]
        self.test_unseen_labels2 = np.ones(self.test_unseen_labels.shape[0]).reshape(-1,1)*2
        self.test_unseen_labels2 = self.test_unseen_labels2.astype('int16')
              
        #self.test_seen_set = self.trainval_set
        #self.test_seen_labels = self.trainval_labels 
  
             
       
        self.train_set = np.vstack([self.train_set_, self.val_set])
        self.train_labels = np.vstack([self.train_labels_, self.val_labels])
        
      
    
        self.seen_labels = np.unique(self.test_seen_labels).astype('int16')
        self.unseen_labels = np.unique(self.test_unseen_labels).astype('int16')
       
        #self.test_unseen_set = np.vstack([self.test_unseen_set, self.test_seen_set])
        #self.test_unseen_labels = np.vstack([self.test_unseen_labels2, self.test_seen_labels2])
    
 

        
    
        
        
    
    
        