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
#from common import general
from sklearn.metrics import pairwise_distances
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from math import factorial
from utilis_svae import emd
from svae_models import classifier
#from svae_models import basis_learning
import torch.nn.functional as F
import scipy.io
import pickle

def norm_data(visual_features):
    for i in range(visual_features.shape[0]):
        visual_features[i,:] = visual_features[i,:]/np.linalg.norm(visual_features[i,:]) 
    return visual_features
    
    
class Model_train(object):
    def __init__(self, 
                 dataset_name,
                 encoder,
                 decoder,
                 attr_encoder,
                 attr_decoder,
                 classifier,
                 train_loader,
                 test_loader_unseen,
                 test_loader_seen,
                 criterion,
                 lr = 1e-3,
                 all_attrs = None,
                 epoch = 10000,
                 save_path = "/data/xingyu/wae_lle/experiments/",
                 save_every = 1,
                 iftest = False,
                 ifsample = False,
                 data = None,
                 GZSL = True,
                 zsl_classifier = None
                 ):  
        self.dataset_name = dataset_name
        self.encoder = encoder
        self.decoder = decoder
        self.attr_encoder = attr_encoder
        self.attr_decoder = attr_decoder
        self.classifier = classifier
        self.zsl_classifier = zsl_classifier
        self.train_loader = train_loader
        self.test_loader_unseen = test_loader_unseen
        self.test_loader_seen = test_loader_seen
           
        self.criterion = criterion
        self.crossEntropy_Loss = nn.NLLLoss()
        
        self.all_attrs = all_attrs
        self.lr = lr
        self.epoch = epoch
        self.save_path = save_path
        self.save_every = save_every
        self.ifsample = ifsample
        self.data = data
        self.GZSL = GZSL
        self.distribution = 'vmf'
        self.sinkhorn = emd.SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        
        if iftest:
            log_dir = '{}/log'.format(self.save_path)
            #general.logger_setup(log_dir, 'results__')
        
        
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
        
        
    
    def compute_acc(self,trues, preds):
        """
        Given true and predicted labels, computes average class-based accuracy.
        """

        # class labels in ground-truth samples
        classes = np.unique(trues)
        # class-based accuracies
        cb_accs = np.zeros(classes.shape, np.float32)
        #ipdb.set_trace()
        for i, label in enumerate(classes):
            inds_ci = np.where(trues == label)[0]

            cb_accs[i] = np.mean(
              np.equal(
              trues[inds_ci],
              preds[inds_ci]
            ).astype(np.float32)
        )
        #ipdb.set_trace()
        return np.mean(cb_accs)   
      
    def training(self, checkpoint = -1):
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
    
        if checkpoint > 0:
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint)
            file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint)
                
            enc_path = os.path.join(self.save_path, file_encoder)
            dec_path = os.path.join(self.save_path, file_decoder)
            attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path, file_attr_decoder)
            classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            classifier_checkpoint = torch.load(classifier_path)
            self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.train()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train()
        
        enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
              
        enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
            
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
                    input_label = input_label.long().view(-1).cuda()
                    input_attr = input_attr.float().cuda().squeeze()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = self.classifier(z_input)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) 
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                
                
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0))
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x.view(-1,1,z_x.shape[1]), z_attr)
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  
            
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict(), 
                     'optimizer': enc_optim.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict(), 
                     'optimizer': dec_optim.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier)   
            
    def search_thres_by_sample(self, attrs, n = 10000):
        min_thres = 100
        m, s = self.attr_encoder(attrs)
      
        z = []
        for i in range(n):
            z_fake = self.reparameterize(m, s).rsample()
            dist = F.cosine_similarity(m, z_fake)
            z.append(z_fake)
            thres = dist.min()
            if min_thres > thres:
                min_thres = thres
        
        return min_thres
        
    def load_models(self, epoch):
        file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
        file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
        file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)  
        file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)  
        enc_path = os.path.join(self.save_path, file_encoder)
        dec_path = os.path.join(self.save_path, file_decoder)
        attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
        classifier_path = os.path.join(self.save_path, file_classifier)
        enc_checkpoint = torch.load(enc_path)
        self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        dec_checkpoint = torch.load(dec_path)
        self.decoder.load_state_dict(dec_checkpoint['state_dict'])
        attr_enc_checkpoint = torch.load(attr_enc_path)
        self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
        classifier_checkpoint = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_checkpoint['state_dict'])       
        
        # Load the ZSL classifiers. These ZSL classifiers can be replaced by any SOTA models! 
        if self.dataset_name == 'AWA1':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/awa1/Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'AWA2':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/awa2/Checkpoint_9_Classifier.pth.tar")
        elif self.dataset_name == 'CUB':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/cub/Checkpoint_7_Classifier.pth.tar")
        elif self.dataset_name == 'FLO':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/flo/Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'SUN':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/sun/Checkpoint_14_Classifier.pth.tar")
        
        self.zsl_classifier.load_state_dict(zsl_classifier_checkpoint['state_dict'])
        
        self.encoder.eval()
        self.decoder.eval()
        self.attr_encoder.eval()  
        self.zsl_classifier.eval()      
        if torch.cuda.is_available():
             self.encoder, self.decoder, self.attr_encoder, self.zsl_classifier, self.classifier = self.encoder.cuda(), self.decoder.cuda(), self.attr_encoder.cuda(), self.zsl_classifier.cuda(), self.classifier.cuda()
    '''    
    def search_thres_by_bases(self, epoch, basis_config, basis_dir,dataset = None, n = 2000):
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        self.load_models(epoch)
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []    
        all_anchors = self.attr_encoder(all_attrs)[0]      
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]       
        seen_count = 0
        seen_all = 1
        unseen_count = 0
        unseen_all = 1
        all_count = 0
        all_z = []
        all_label = []
        D = []
        
        thres = np.zeros(seen_labels.shape[0])
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            #z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            all_z.append(z_real.cpu().data)
            all_label.append(input_label.cpu().data)
        
        all_real = np.vstack(all_z)
        all_real_label = np.vstack(all_label)
        
        for i in range(seen_labels.shape[0]):
            class_id = seen_labels[i]
            idx = np.where(all_real_label == class_id)[0]
            data_i_real = all_real[idx,:]
            m, s = self.attr_encoder(all_attrs[class_id-1,:])
            z = self.reparameterize(m, s) 
            data_i_fake = z.rsample(2000).cpu().data
            
            data_all = np.vstack([data_i_real, data_i_fake])            
            B = basis_learning.Basis_learning(
                    basis_config, 
                    data_all, 
                    data_all[0:10,:], 
                    basis_dir)
                 
            z_tile = m.repeat(basis_config['basis_num']).view(basis_config['basis_num'],-1)
            dist = F.cosine_similarity(z_tile, torch.Tensor(B.D.T).cuda())      
            thres[i] = dist.max()  
            D.append(B.D)            
            
        with open(os.path.join(basis_dir, "res.pkl"), 'wb') as f:      
            pickle.dump({'D': D}, f)  
            f.close()
            print('save bases at {}'.format(basis_dir))

        return thres
    '''    
    
    def search_thres_by_traindata(self, epoch, dataset = None, n = 0.95):
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        self.load_models(epoch)

        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []    
        all_anchors = self.attr_encoder(all_attrs)[0]      
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]       
        seen_count = 0
        seen_all = 0
        unseen_count = 0
        unseen_all = 0
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            #z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                kk = input_label[k,:]+1
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                if min_thres>dist.max():
                    min_thres = dist.max()
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
            
        dist_array = np.array(dist_list)
        idx = dist_array.shape[0] * (1.0 - n)
        thres  = np.sort(dist_array)[int(idx)]

      
        return thres 
              
    def testing_2(self, epoch, test_class = 'seen', dataset = None, threshold = 0.99):
        
        if test_class == 'seen':
            test_loader = self.test_loader_seen
        elif test_class == 'unseen':
            test_loader = self.test_loader_unseen
        
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        else:
            thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        
        seen_count = 0
        seen_all = 1
        unseen_count = 0
        unseen_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []
        for i_batch, sample_batched in enumerate(test_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]           
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]+1
                gt.append(kk.data.item()-1)
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        out = self.zsl_classifier(input_k.view(1,-1))
                        pred_label_ = torch.argmax(out,1)
                        pred_label = self.data.unseen_labels[pred_label_.cpu().data.item()]-1
                        pred.append(pred_label)
                        unseen_count +=1
                    else:
                        pred.append(1000)
                    
                    
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1            
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1
                        out = self.classifier(z_real[k,:].view(1,-1))
                        pred_label = torch.argmax(out,1).data.item()
                        #pred_label = self.data.test_seen_labels[pred_label_.cpu().data.item()]-1
                        pred.append(pred_label)
                    else:
                        pred.append(1000)
               
                                        
        pred_ = np.vstack(pred)
        gt_ = np.vstack(gt)
        acc = self.compute_acc(gt_, pred_ )

        mean_dist = mean_dist /all_count
        return unseen_count/unseen_all, seen_count/seen_all , acc, dist_list
        
    def draw_roc_curve(self, epoch, data):
        import sklearn.metrics as metrics
        unseen_acc, _, ts, dist_unseen = self.testing_2(epoch, test_class ='unseen', dataset = data, threshold = 0.63)
        _, seen_acc,  tr, dist_seen = self.testing_2(epoch, test_class ='seen', dataset = data, threshold = 0.63)
         
        print('fpr = {}, tpr = {}'.format(1-unseen_acc, seen_acc)) 
        ipdb.set_trace()
        dists = np.concatenate((np.array(dist_unseen), np.array(dist_seen)))
        
        labels_unseen = np.zeros(len(dist_unseen))
        labels_seen = np.ones(len(dist_seen))
        
        labels = np.concatenate((labels_unseen, labels_seen))
        fpr, tpr, threshold = metrics.roc_curve(labels, dists)
        roc_auc = metrics.auc(fpr, tpr)
        
        #print('fpr = {}, tpr = {}, auc = {}'.format(1-fpr, tpr, roc_auc))
        
        with open("{}_res.pkl".format(self.dataset_name), 'wb') as f:      
            pickle.dump({'fpr': fpr, 'tpr':tpr}, f)  
            f.close()
            print('save data done!')
            
        plt.title('ROC curves on the 5 benchmark datasets')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        ipdb.set_trace()
        return 0 
    
    def testing(self, epoch, if_viz = True, sample_rate = 2):
        file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
        file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
        file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
        
        enc_path = os.path.join(self.save_path, file_encoder)
        dec_path = os.path.join(self.save_path, file_decoder)
        attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
    
        enc_checkpoint = torch.load(enc_path)
        self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
        dec_checkpoint = torch.load(dec_path)
        self.decoder.load_state_dict(dec_checkpoint['state_dict'])
        
        attr_enc_checkpoint = torch.load(attr_enc_path)
        self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
         
        self.encoder.eval()
        self.decoder.eval()
        self.attr_encoder.eval()
        
        if torch.cuda.is_available():
             self.encoder, self.decoder, self.attr_encoder = self.encoder.cuda(), self.decoder.cuda(), self.attr_encoder.cuda()
             
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        '''
        class_names = ["antelope", "grizzly bear", "killer whale", "beaver", "dalmatian", "persian cat", "horse",
                           "german shepherd", "blue whale", "siamese cat", "skunk",  "mole", "tiger", "hippopotamus",
                           "leopard", "moose", "spider monkey", "humpback whale", "elephant", "gorilla", "ox",  "fox",
                           "sheep", "seal" ,"chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat", "giraffe",
                           "wolf", "chihuahua", "rat", "weasel","otter", "buffalo", "zebra", "giant panda", "deer", "bobcat",
                           "pig", "lion", "mouse", "polar bear", "collie", "walrus", "raccoon", "cow", "dolphin"]
        '''
        class_names = ["Seen Features","Unseen Features"]                   
        
        for i_batch, sample_batched in enumerate(self.test_loader_unseen):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]
            
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()
            
            
            if self.ifsample:
                m, s = self.encoder(input_data)
                z_real = self.reparametrize(m, s)
            else:
                z_real = self.encoder(input_data)[0]
            
            x_recon = self.decoder(z_real)
                
            mu, sigma = self.attr_encoder(input_attr)
            z_fake = self.reparameterize(mu, sigma).rsample().squeeze()
       
            muu.append(z_fake.squeeze().cpu().data.numpy())
            
            z.append(z_real.cpu().data.numpy())
            label.append(input_label.cpu().data.numpy().reshape(-1,1))
            recon.append(x_recon.cpu().data.numpy())
            data_in.append(input_data.cpu().data.numpy())
            z_attr.append(z_fake.squeeze().cpu().data.numpy())
            
            recon_loss = self.criterion(x_recon, input_data)
            recon_loss = torch.dot(z_real[1,:], z_fake[1,:])
            print('batch {} recon_loss = {}'.format(i_batch, recon_loss))
        
   
        muu_ = np.vstack(muu)      
        z_ = np.vstack(z)
        recon_ = np.vstack(recon)
        label_ = np.vstack(label).reshape(-1)
        data_in_ = np.vstack(data_in)
        z_attr_ = np.vstack(z_attr)
      
        if if_viz:
            from sklearn.manifold import TSNE
            from matplotlib import colors as mcolors

            colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
            color_list = []
            for color in colors.keys():
                if color == 'aliceblue':
                    color_list.append('y')
                elif color == 'k':
                    color_list.append('purple')
                else:
                    color_list.append(color)
            
            color_list[0] = 'blue'
            color_list[1] = 'darkorange'
            
            label_colors = []
            label_names = []
     
            for i in range(len(z_attr_)):
                label_colors.append(color_list[label_[i]])
                label_names.append(class_names[label_[i]])
          
        
            model = TSNE(n_components = 2, n_iter = 5000, init = 'pca',random_state = 0)
               
            #zz_ = np.vstack([z_, muu_])
            zz_ = np.vstack([z_, z_])
            label_colors__ = label_colors
            label_colors = label_colors + label_colors__      
            z_sample = zz_[range(0,zz_.shape[0],sample_rate),:]
            label_colors_sample = label_colors[::sample_rate] 
            label_names_sample = label_names[::sample_rate] 

            z_2d = model.fit_transform(z_sample)
            fig = plt.figure(figsize = (12, 12) )
            ax = fig.add_subplot(111)
            n = z_2d.shape[0]
            
            
            df1 = pd.DataFrame({"x": z_2d[0:n//2, 0], "y": z_2d[0:n//2, 1], "colors": label_colors_sample[0:n//2]})
            for i, dff in df1.groupby("colors"):
                class_name = class_names[color_list.index(i)]
                plt.scatter(dff['x'], dff['y'], c=i, label= class_name, marker = '.')
              
            ax.scatter(z_2d[0:n//2, 0], z_2d[0:n//2, 1], c=label_colors_sample[0:n//2] , marker = '.')
            ax.scatter(z_2d[n//2:n, 0], z_2d[n//2:n, 1], c=label_colors_sample[n//2:n], marker = '.')
            ax.set_facecolor('gray')
            #ax.set_ylim(-48, 48)
            #ax.set_xlim(-48, 48)
            plt.axis('off')
            #ax.set_yticklabels([])
            #ax.set_xticklabels([])
            
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
            ax.legend(fontsize = 'xx-large',loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
            #plt.legend(fontsize = "small", loc=1)
            plt.show()
            ipdb.set_trace()
        return z_, recon_, label        
            
            
