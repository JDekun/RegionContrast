
import torch.nn as nn
from torch.nn import functional as F
import importlib
from .base import Aux_Module



class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg['sync_bn']
        self._num_classes = net_cfg['num_classes']

        self.encoder = self._build_encoder(net_cfg['encoder'])
        self.decoder = self._build_decoder(net_cfg['decoder'])

        self._use_auxloss = True if net_cfg.get('aux_loss', False) else False
        self.fpn = True if net_cfg['encoder']["kwargs"].get('fpn', False) else False
        self.contrast = True if 'contrast' in net_cfg['decoder']['type'] else False
        if self._use_auxloss:
            cfg_aux = net_cfg['aux_loss']
            self.loss_weight = cfg_aux['loss_weight']
            self.auxor = Aux_Module(cfg_aux['aux_plane'], self._num_classes, self._sync_bn)

    def _build_encoder(self, enc_cfg):
        enc_cfg['kwargs'].update({'sync_bn': self._sync_bn})
        encoder = self._build_module(enc_cfg['type'], enc_cfg['kwargs'])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg['kwargs'].update({'in_planes': self.encoder.get_outplanes(),
                                  'sync_bn': self._sync_bn,
                                  'num_classes': self._num_classes})
        decoder = self._build_module(dec_cfg['type'], dec_cfg['kwargs'])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x, infer=False, is_eval=False):
        n, c, h, w = x.size()
        if self._use_auxloss:
            if self.fpn and infer:
                f1, f2, feat1, feat2 = self.encoder(x)
                pred_head = self.decoder([f1, f2,feat1, feat2], infer=True, is_eval = is_eval)
            elif self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x)
                pred_head = self.decoder([f1, f2,feat1, feat2], is_eval = is_eval)
            else:
                feat1, feat2 = self.encoder(x)
                pred_head = self.decoder(feat2, is_eval = is_eval)

            if not is_eval:
                pred_aux = self.auxor(feat1)
                pred_aux = F.interpolate(input=pred_aux, size=(h, w), mode='bilinear', align_corners=True)
            
            if self.contrast:
                if not is_eval:
                    res, contrast_loss = pred_head
                    res = F.interpolate(input=res, size=(h, w), mode='bilinear', align_corners=True)
                    return [res, pred_aux, contrast_loss]
                else:
                    res = pred_head
                    res = F.interpolate(input=res, size=(h, w), mode='bilinear', align_corners=True)
                    #fea = F.interpolate(input=fea, size=(h, w), mode='bilinear', align_corners=True)
                    return [res]            
            else:
                pred_head = F.interpolate(input=pred_head, size=(h, w), mode='bilinear', align_corners=True)
                if not is_eval:
                    return [pred_head, pred_aux]
                else:
                    return [pred_head]
        else:
            feat = self.encoder(x)
            pred_head = self.decoder(feat)
            pred_head = F.interpolate(input=pred_head, size=(h, w), mode='bilinear', align_corners=True)
            return pred_head