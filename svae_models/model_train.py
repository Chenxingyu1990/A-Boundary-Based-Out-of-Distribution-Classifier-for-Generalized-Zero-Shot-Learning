import torch
import ipdb
import logging
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from common import general
#from utils import mmd_utils
from sklearn.metrics import pairwise_distances
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
#from wae_models import classifier
from math import factorial
from utilis_svae import emd

def norm_data(visual_features):
    for i in range(visual_features.shape[0]):
        visual_features[i,:] = visual_features[i,:]/np.linalg.norm(visual_features[i,:]) 
    return visual_features
    
    
class Model_train(object):
    def __init__(self, 
                 encoder,
                 decoder,
                 attr_encoder,
                 attr_decoder,
                 classifier,
                 train_loader,
                 test_loader_unseen,
                 test_loader_seen,
                 criterion,
                 SIGMA = 2,
                 lr = 1e-3,
                 all_attrs = None,
                 epoch = 10000,
                 save_path = "/data/xingyu/wae_lle/experiments/",
                 save_every = 1,
                 iftest = False,
                 ifsample = False,
                 data = None,
                 GZSL = True
                 ):  
        self.encoder = encoder
        self.decoder = decoder
        self.attr_encoder = attr_encoder
        self.attr_decoder = attr_decoder
        self.classifier = classifier
        self.train_loader = train_loader
        self.test_loader_unseen = test_loader_unseen
        self.test_loader_seen = test_loader_seen
           
        self.criterion = criterion
        self.crossEntropy_Loss = nn.NLLLoss()
        
        self.all_attrs = all_attrs
        self.lr = lr
        self.epoch = epoch
        self.SIGMA = SIGMA
        self.save_path = save_path
        self.save_every = save_every
        self.ifsample = ifsample
        self.data = data
        self.GZSL = GZSL
        self.distribution = 'vmf'
        self.sinkhorn = emd.SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        
        if iftest:
            log_dir = '{}/log'.format(self.save_path)
            general.logger_setup(log_dir, 'results__')
        
        
    def save_checkpoint(self,state, filename = 'checkpoint.pth.tar'):
        torch.save(state, filename)  
         
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
        else:
            raise NotImplemented

        return q_z
        
    def KL_divergence(self, u_q, k_q, u_p, k_p):
    
        def calculate_term2(k_q, d):
            res = k_q.clone()
            k_q_tmp = k_q.clone()
            for i in range(1,int(d)):
                m = i + 1
                xx = k_q_tmp.pow(m)
                res_m =  xx / factorial(m)   
                res += res_m
                if res_m.mean() < 1e-8 or res_m.mean() > 1e8:
                    return res.log()
            return res.log()
                 
        u_q = u_q.t()
        k_q = k_q.t()
        u_p = u_p.squeeze().t()
        k_p = k_p.squeeze().t().view(1, -1)
        
        d = u_p.shape[0]
        d1 = (d -1)/2
        d2 = (d -3)/2

        term_1 = k_q - torch.mm(torch.mm(k_p, u_p.t()), u_q) + d1 * k_q.log() - d1 * k_p.log()
        term_2 = calculate_term2(k_q, 5)
        term_3 = calculate_term2(k_p, 5)           
        kl_divergence = term_1 - term_2 + term_3

        #print("term1 {} term2 {} term3 {} k_p {} k_q {}".format(term_1.mean(), term_2.mean(), term_3.mean(), k_p.mean(), k_q.mean()))
        return kl_divergence.mean().abs()
        
    def training(self, checkpoint = -1):
        log_dir = '{}/log'.format(self.save_path)
        general.logger_setup(log_dir)
        
        if checkpoint >= 0:
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint)
                
            enc_path = os.path.join(self.save_path, file_encoder)
            dec_path = os.path.join(self.save_path, file_decoder)
            attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path, file_attr_decoder)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
                
        self.encoder.train()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_decoder.train()
        
        enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
              
        enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
        
        for epoch in range(checkpoint+1, self.epoch):
            step = 0            
            train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.cuda()
                    input_attr = input_attr.float().cuda()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
                
                attr_fake = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_fake, input_attr)
           
                #KL_loss = self.KL_divergence(m1, s1, m2, s2)
                
                #KL_loss = torch.distributions.kl.kl_divergence(z1, z2).mean()
               
                dist, P, C = self.sinkhorn(z_x.view(-1,1,64), z_attr)
   
                #print(KL_loss)
                KL_loss = dist.mean()
                total_loss =  recon_loss *1.0 + KL_loss * 0.01  + attr_loss *1.0 
                
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    logging.info("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
            
            
            
            
            
            
            