import numpy as np
import torch
import os
import torch.nn as nn
import ipdb
import yaml
import argparse
from shutil import copyfile
from utilis_svae import datasets
from svae_models import model_train
from torch.utils.data import Dataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('template_path', type=str)
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    with open(args.template_path, 'r') as f:
        template = yaml.load(f)
    dataset_config = template['dataset'] 
    model_config = template['model']
    train_config = template['train']
    save_path = '{}_{}_{}'.format(dataset_config['save_path'],model_config['mid_size'], model_config['hidden_size'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    config_dir = os.path.join(save_path, 'config')
    log_dir = os.path.join(save_path, 'log')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
        
    config_copy = '{}/{}'.format(config_dir, args.template_path)
    copyfile(args.template_path, config_copy)
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    '''    
    basis_dir = '{}/basis'.format(train_config['res_dir'])
    if not os.path.exists(basis_dir):
        os.mkdir(basis_dir)
    '''
   
    if dataset_config['dataset_name'] == 'ImageNet':
        data = datasets.Imagenet(dataset_config['dataset_name'],
                            dataset_config['data_path'],
                            iftrain = True,
                            n_neighbors = dataset_config['n_neighbors'])  
    else:
     
        data = datasets.AwA2(dataset_config['dataset_name'],
                            dataset_config['data_path'],
                            iftrain = True,
                            n_neighbors = dataset_config['n_neighbors'])                      
    
    
    dataset_setup_train = datasets.Dataset_setup(
                            data = data.train_set,
                            attrs = data.attrs ,
                            labels = data.train_labels 
                            )
        
    dataset_setup_val = datasets.Dataset_setup(
                            data = data.val_set,
                            attrs = data.attrs ,
                            labels = data.val_labels 
                            )
    dataset_setup_test_seen = datasets.Dataset_setup(
                            data = data.test_seen_set,
                            attrs = data.attrs ,
                            labels = data.test_seen_labels 
                            )
       
    dataset_setup_test_unseen = datasets.Dataset_setup(
                            data = data.test_unseen_set,
                            attrs = data.attrs ,
                            labels = data.test_unseen_labels 
                            )

  
    
    
    
    dataset_loader_train = torch.utils.data.DataLoader(dataset_setup_train, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    dataset_loader_val = torch.utils.data.DataLoader(dataset_setup_val, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    dataset_loader_test_seen = torch.utils.data.DataLoader(dataset_setup_test_seen, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    dataset_loader_test_unseen = torch.utils.data.DataLoader(dataset_setup_test_unseen, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    
    
    
    from svae_models import models
        
    attr_encoder = models.Attr_Encoder(model_config['attr_size'], model_config['mid_size'], model_config['hidden_size'])
    attr_decoder = models.Attr_Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['attr_size'])
    if model_config['ifsample']:
        encoder = models.Encoder2(model_config['input_size'], model_config['mid_size'], model_config['hidden_size'])
    else:
        encoder = models.Encoder(model_config['input_size'], model_config['mid_size'], model_config['hidden_size'])
    
    
    decoder = models.Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['input_size'])

        
    from svae_models import model_train
    
    if dataset_config['GZSL']:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], model_config['classes'])
    else:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], data.unseen_labels.shape[0])
    print(attr_encoder)
    print(attr_decoder)
    print(encoder)
    print(decoder)
    print(classifier)

   
    model_train_obj = model_train.Model_train(
                                  dataset_config['dataset_name'],
                                  encoder,
                                  decoder,
                                  attr_encoder,
                                  attr_decoder,
                                  classifier,
                                  dataset_loader_train,
                                  dataset_loader_test_unseen,
                                  dataset_loader_test_seen,
                                  criterion = nn.L1Loss(),
                                  SIGMA = 0.5,
                                  lr = 1e-4,
                                  all_attrs = data.attrs,
                                  epoch = train_config['epoch'],
                                  save_path = save_path,
                                  save_every = train_config['save_every'],
                                  ifsample = model_config['ifsample'],
                                  data = data,
                                  GZSL = dataset_config['GZSL']
                                  )
                                  
    model_train_obj.training(train_config['check_point'])
    '''
    for i in range(train_config['start_epoch_classifier'],train_config['epoch'],5):
        model_train_obj.training_classifier(epoch = i, classifier_epoch = train_config['classifier_epoch'], n = train_config['generate_n_samples'])
    '''