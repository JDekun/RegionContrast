import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import  ASPP, get_syncbn

class dec_deeplabv3_contrast(nn.Module):
   
    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), temperature=0.2, queue_len=2975):
        super(dec_deeplabv3_contrast, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.temperature = temperature
        self.queue_len = queue_len
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))
        self.final =  nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        for i in range(num_classes):
            self.register_buffer("queue"+str(i),torch.randn(inner_planes,self.queue_len))
            self.register_buffer("ptr"+str(i),torch.zeros(1,dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')
           
    def _dequeue_and_enqueue(self,keys,vals,cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        eval("self.queue"+str(cat))[:,ptr] = keys
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr

    def construct_region(self, fea, pred):
        bs = fea.shape[0]
        pred = pred.max(1)[1].squeeze().view(bs, -1)  
        val = torch.unique(pred)
        fea=fea.squeeze()
        fea = fea.view(bs, 256,-1).permute(1,0,2) 
    
        new_fea = fea[:,pred==val[0]].mean(1).unsqueeze(0) 
        for i in val[1:]:
            if(i<19):
                class_fea = fea[:,pred==i].mean(1).unsqueeze(0)   
                new_fea = torch.cat((new_fea,class_fea),dim=0)
        val = torch.tensor([i for i in val if i<19])
        return new_fea, val.cuda()

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(1)
        l_pos = l_pos.transpose(0,1)
        l_neg = l_neg.repeat(N,1)
        logits = torch.cat((l_pos,l_neg),dim=1) #256, N1*2
        logits /= self.temperature
        labels = torch.zeros((N,),dtype=torch.long).cuda()
        return self.criterion(logits,labels)
    

    def forward(self, x, is_eval = False):
     
        aspp_out = self.aspp(x)
        fea = self.head(aspp_out)
        res = self.final(fea)
        if not is_eval:
            bs = x.shape[0]
            keys, vals = self.construct_region(fea, res)  #keys: N,256   vals: N,  N is the category number in this batch
            keys = nn.functional.normalize(keys,dim=1)
            contrast_loss = 0

            for cls_ind in range(self.num_classes):
                if cls_ind in vals:
                    query = keys[list(vals).index(cls_ind)]   #256,
                    l_pos = query.unsqueeze(1)*eval("self.queue"+str(cls_ind)).clone().detach()  #256, N1
                    l_pos = l_pos.mean(0).unsqueeze(0)
                    all_ind = [m for m in range(19)]
                    l_neg = 0
                    tmp = all_ind.copy()
                    tmp.remove(cls_ind)
                    i = 0
                    for cls_ind2 in tmp:
                        temp = query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).clone().detach() #256, N1
                        temp = temp.mean(0).unsqueeze(0)
                        if i != 0:
                            l_neg = torch.cat((l_neg, temp),dim=1)
                        else:
                            l_neg = temp
                        i += 1
                    contrast_loss += self._compute_contrast_loss(l_pos, l_neg)
                else:
                    continue
            for i in range(self.num_classes):
                self._dequeue_and_enqueue(keys,vals,i, 1)
            return res, contrast_loss
        return res
