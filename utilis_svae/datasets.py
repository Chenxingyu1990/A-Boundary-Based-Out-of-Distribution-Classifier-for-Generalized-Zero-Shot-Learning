import numpy as np
import torch
import scipy.io
import os
import ipdb
import pickle
import h5py
from utils import LLE_utils
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
        attr_idx = self.labels[idx].astype('uint8') -1
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
        attr_idx = self.labels[idx].astype('uint8') -1
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
            self.attrs[i,:] = self.attrs[i,:]/np.linalg.norm(self.attrs[i,:])* 10.0
        print('norm attributes done!')
    
        for i in range(self.visual_features.shape[0]):
            self.visual_features[i,:] = self.visual_features[i,:]/np.linalg.norm(self.visual_features[i,:]) 
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
            features = f['features'][:].T
            features_val = f['features_val'][:]
            labels = f['labels'][:]
            labels_val = f['labels_val'][:]     
            self.visual_features = np.vstack([features, features_val])
            self.visual_labels = np.hstack([labels, labels_val])     
                    
        if self.ifnorm:
            self.norm_data()
        
        self.train_set = self.visual_features
        self.train_labels = self.visual_labels.T
        self.seen_labels = np.unique(self.train_labels).astype('uint8')
        self.unseen_labels = np.unique(self.train_labels).astype('uint8')
        if self.iftrain:
            train_set_pkl_path = os.path.join(self.data_path, "train_set_{}.pkl".format(self.n_neighbors))
            test_set_pkl_path = os.path.join(self.data_path, "test_set_{}.pkl".format(self.n_neighbors))
            if not os.path.exists(train_set_pkl_path):
                self.train_set, self.train_labels = KNN_utils.batch_mmd(self.train_set, self.train_labels, self.n_neighbors)
                with open(train_set_pkl_path, 'wb') as f:      
                    pickle.dump({'train_set': self.train_set, 'train_labels': self.train_labels}, f)  
                    f.close()
                    print('save train set at {}'.format(train_set_pkl_path))
            else:
                with open(train_set_pkl_path, 'rb') as f:
                    train__ = pickle.load(f)
                    f.close()
                self.train_set = train__['train_set']
                self.train_labels = train__['train_labels']

        
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
            self.visual_features[i,:] = self.visual_features[i,:]/np.linalg.norm(self.visual_features[i,:]) * 10.0
        print('norm features done!')
        
    def prepare_data(self):
        
        feature_path = os.path.join(self.data_path, "res101.mat")
        attr_path = os.path.join(self.data_path, "att_splits.mat")
        
        features = scipy.io.loadmat(feature_path)
        attr = scipy.io.loadmat(attr_path)
        self.visual_features = features['features'].T
        self.visual_labels = features['labels']
        self.attrs = attr['att'].T 
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
        
        
        
        self.test_seen_set = self.visual_features[self.test_seen_loc.reshape(-1)-1,:]
        self.test_seen_labels = self.visual_labels[self.test_seen_loc.reshape(-1)-1,:]
        
        self.test_unseen_set = self.visual_features[self.test_unseen_loc.reshape(-1)-1,:]
        self.test_unseen_labels = self.visual_labels[self.test_unseen_loc.reshape(-1)-1,:]
        
        self.seen_labels = np.unique(self.test_seen_labels).astype('uint8')
        self.unseen_labels = np.unique(self.test_unseen_labels).astype('uint8')
  
        self.train_set = np.vstack([self.train_set_, self.val_set])
        self.train_labels = np.vstack([self.train_labels_, self.val_labels])
        
        self.test_set = np.vstack([self.test_unseen_set, self.test_seen_set])
        self.test_labels = np.vstack([self.test_unseen_labels, self.test_seen_labels])
        '''
        ipdb.set_trace()
        if self.iftrain:
            train_set_pkl_path = os.path.join(self.data_path, "train_set_{}.pkl".format(self.n_neighbors))
            test_set_pkl_path = os.path.join(self.data_path, "test_set_{}.pkl".format(self.n_neighbors))
            if not os.path.exists(train_set_pkl_path):
                self.train_set, self.train_labels = KNN_utils.batch_mmd_KNN(self.train_set, self.train_labels, self.n_neighbors)
                with open(train_set_pkl_path, 'wb') as f:      
                    pickle.dump({'train_set': self.train_set, 'train_labels': self.train_labels}, f)  
                    f.close()
                    print('save train set at {}'.format(train_set_pkl_path))
            else:
                with open(train_set_pkl_path, 'rb') as f:
                    train__ = pickle.load(f)
                    f.close()
                self.train_set = train__['train_set']
                self.train_labels = train__['train_labels']
                    
            if not os.path.exists(test_set_pkl_path):
                self.test_unseen_set, self.test_unseen_labels = KNN_utils.batch_mmd_KNN(self.test_unseen_set, self.test_unseen_labels, self.n_neighbors)
                with open(test_set_pkl_path, 'wb') as f: 
                    pickle.dump({'test_set': self.test_unseen_set, 'test_labels': self.test_unseen_labels}, f)  
                    f.close()
                    print('save test set at {}'.format(test_set_pkl_path))
            else:
                with open(test_set_pkl_path, 'rb') as f:
                    test__ = pickle.load(f)
                    f.close()
                self.test_unseen_set = test__['test_set']
                self.test_unseen_labels = test__['test_labels']
            #self.val_set, self.val_labels = KNN_utils.batch_mmd_KNN(self.val_set, self.val_labels, 30)
            #self.test_seen_set, self.test_seen_labels = KNN_utils.batch_mmd_KNN(self.test_seen_set, self.test_unseen_labels, 30)
            #self.test_unseen_set, self.test_unseen_labels = KNN_utils.batch_mmd_KNN(self.test_unseen_set, self.test_unseen_labels, self.n_neighbors)
        '''
 

        
    


class Swiss_roll(object):
    def __init__(self,
                 N,
                 noise = 0.0):  
        self.N = N
        self.prepare_data()  
    
    def prepare_data(self):
        train_set, color = make_swiss_roll(self.N, self.noise)
        test_set = train_set
        
        self.train_data = {"data": train_set, "label": color}
        self.test_data = {"data": test_set, "label":color}



        
    '''    
    def __len__(self):
        return self.visual_labels.shape[0]
    
    def __getitem__(self, idx):
        sample_idx = self.visual_features[:, idx]
        attr_idx = self.visual_labels[idx]
        attr = self.attrs[:, attr_idx-1]
        sample = {'feature': sample_idx, 'attr': attr, 'label': attr_idx}
        
        return sample
    '''
        
        
    
    
        