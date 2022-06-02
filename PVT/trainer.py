from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy

class trainer():
     def __init__(self,train_iter,test_iter,model,optimizer,device,scheduler):
         self.train_iter = train_iter
         self.test_iter = test_iter 
         self.optimizer = optimizer
         self.scheduler = scheduler
         self.device = device
         self.model = model.to(device)
         self.loss = LabelSmoothingCrossEntropy()
     
     def train(self,num_epoch,alg,param_dir=None):
         self.model.train()
         writer = SummaryWriter("tensorboard/"+alg)
         iteration = 0
         train_loss = []
         test_loss = []
         test_acc = []
         for i in range(1,1+num_epoch):
             for X,y in self.train_iter:
                 X = X.to(self.device)
                 y = y.to(self.device)
                 y_hat = self.model(X)
                 l = self.loss(y_hat,y)
                 self.optimizer.zero_grad()
                 l.backward()
                 self.optimizer.step()
                 
                 train_loss.append(l.item())
                 acc, l_test = self.test()
                 test_loss.append(l_test)
                 test_acc.append(acc)
                 iteration+=1
                 writer.add_scalar("Train_loss", l, iteration)
                 writer.add_scalar("Test_loss", l_test, iteration)
                 writer.add_scalar("Test_accuracy", acc, iteration)
                 
                 
             self.scheduler.step()
             
             if i%5==0:
                 print('epoch:{}, train_loss:{}, test_loss:{}, test_accuracy:{}'.format(i,l,l_test,acc))
         
         writer.close()
         if param_dir is not None:
             self.model.cpu()
             torch.save(self.model.state_dict(),param_dir) 
                
         return train_loss,test_loss,test_acc
     
         
     def test(self):
         self.model.eval()
         l = 0
         n = 0
         acc = 0
         with torch.no_grad():
             for X,y in self.test_iter:
                 X = X.to(self.device)
                 y = y.to(self.device)
                 y_hat = self.model(X)
                 l += (self.loss(y_hat,y)*len(y)).item()
                 n += len(y)
                 acc += (y_hat.argmax(dim = 1) == y).float().sum().item() 
         self.model.train()
         return acc/n,l/n