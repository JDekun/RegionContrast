import argparse
import os.path as osp
import yaml
import logging
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn

from pyseg.models.model_helper import ModelBuilder

import torch.distributed as dist

from pyseg.utils.loss_helper import get_criterion
from pyseg.utils.lr_helper import get_scheduler, get_optimizer

from pyseg.utils.utils import AverageMeter, intersectionAndUnion, init_log, load_trained_model
from pyseg.utils.utils import set_random_seed, get_world_size, get_rank
from pyseg.dataset.builder import get_loader

parser = argparse.ArgumentParser(description="Pytorch Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument("--local_rank", type=int, default=0)


# 添加的参数
parser.add_argument("--save_dir", type=str, default="default")
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--batch_size_val', default=2, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument("--amp",dest="amp",action="store_true")
parser.add_argument("--resume",type=str, default="")

logger =init_log('global', logging.INFO)
logger.propagate = 0


def main():
    global args, cfg
    args = parser.parse_args()

    cfg = yaml.load(open("./scripts/"+args.config, 'r'), Loader=yaml.Loader)

    cfg['dataset'].update({'batch_size': args.batch_size, 'batch_size_val': args.batch_size_val})
    cfg['trainer'].update({'epochs': args.epochs})
    
    cudnn.enabled = True
    cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        #synchronize()
    rank = get_rank()
    world_size = get_world_size()
    print('rank,world_size',rank,world_size)
    if rank == 0:
        logger.info(cfg)
    if args.seed is not None:
        print('set random seed to',args.seed)
        set_random_seed(args.seed)

    save_dir = cfg['saver']['snapshot_dir'] +"/" + args.save_dir

    if not osp.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    # 保存mIOU
    results_csv = save_dir + "/metadata.csv"
    if rank == 0:
        # write into csv
        with open(results_csv, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"epoch,mIOU,best_mIOU\n" 
            f.write(train_info)

    # amp设置
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Create network.
    model = ModelBuilder(cfg['net'])
    modules_back = [model.encoder]
    modules_head = [model.auxor, model.decoder]
   
    device = torch.device("cuda")
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            find_unused_parameters=False,
        ) 
    if cfg['saver']['pretrain']:
        state_dict = torch.load(cfg['saver']['pretrain'], map_location='cpu')['model_state']
        print("Load trained model from ", str(cfg['saver']['pretrain']))
        load_trained_model(model, state_dict)
    
    #model.cuda()
    if rank ==0:
        logger.info(model)

    criterion = get_criterion(cfg)

    trainloader, valloader = get_loader(cfg)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg['trainer']
    cfg_optim = cfg_trainer['optimizer']

    params_list = []
    for module in modules_back:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr']))
    for module in modules_head:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr']*10))

    optimizer = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(cfg_trainer, len(trainloader), optimizer)  # TODO

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load("../../input/resume/"+args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        # if not args.run_id and ('run_id' in checkpoint.keys()):
        #     args.run_id = checkpoint['run_id']
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        
    # Start to train model
    best_prec = 0
    for epoch in range(start_epoch, cfg_trainer['epochs']):
        # Training
        train(model, optimizer, lr_scheduler, criterion, trainloader, epoch, scaler)
        # Validataion
        if cfg_trainer["eval_on"]:
            # if rank == 0:
            #     logger.info("start evaluation")
            prec = validate(model, valloader, epoch)
            if rank == 0:
                # write into txt
                with open(results_csv, "a") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    train_info = f"{epoch},{prec},{best_prec}\n" 
                    f.write(train_info)
            if rank == 0:
                if prec > best_prec:
                    best_prec = prec
                    state = {'epoch': epoch,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict()}
                    if args.amp:
                        state["scaler"] = scaler.state_dict()
                    torch.save(state, osp.join(save_dir, 'best.pth'))
                logger.info('the best val result is: {}'.format(best_prec))
        # note we also save the last epoch checkpoint
        if epoch == (cfg_trainer['epochs'] - 1) and rank == 0:
            state = {'epoch': epoch,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, osp.join(save_dir, 'epoch_' + str(epoch) + '.pth'))
            logger.info('Save Checkpoint {}'.format(epoch))
    


def train(model, optimizer, lr_scheduler, criterion, data_loader, epoch, scaler):
    model.train()
    data_loader.sampler.set_epoch(epoch)
    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']
    rank, world_size = get_rank(), get_world_size()

    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if rank == 0:
        logger.info('=========epoch[{}]=========,Train'.format(epoch))

    for step, batch in enumerate(data_loader):
        i_iter = epoch * len(data_loader) + step
        lr = lr_scheduler.get_lr()
        lr_scheduler.step()

        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.cuda.amp.autocast(enabled= scaler is not None):
            preds = model(images, is_eval=False)
            if len(preds)>2:
                contrast_loss = preds[-1] / world_size
                loss = criterion(preds[:-1], labels) / world_size
                loss += cfg['criterion']['contrast_weight']*contrast_loss
            else:
                loss = criterion(preds[:], labels) / world_size
            
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # # get the output produced by model
        # output = preds[0] if cfg['net'].get('aux_loss', False) else preds
        # output = output.data.max(1)[1].cpu().numpy()
        # target = labels.cpu().numpy()
       
        # # start to calculate miou
        # intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        # # gather all validation information
        # reduced_intersection = torch.from_numpy(intersection).cuda()
        # reduced_union = torch.from_numpy(union).cuda()
        # reduced_target = torch.from_numpy(target).cuda()

        # dist.all_reduce(reduced_intersection)
        # dist.all_reduce(reduced_union)
        # dist.all_reduce(reduced_target)

        # intersection_meter.update(reduced_intersection.cpu().numpy())
        # union_meter.update(reduced_union.cpu().numpy())
        # target_meter.update(reduced_target.cpu().numpy())

        # gather all loss from different gpus
        # reduced_loss = loss.clone()
        # dist.all_reduce(reduced_loss)
        # #print('rank,reduced_loss',rank,reduced_loss)
        losses.update(loss.item())

        if i_iter % round(50/args.batch_size) == 0 and rank==0:
            # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            # mIoU = np.mean(iou_class)
            # mAcc = np.mean(accuracy_class)
            logger.info('iter = {} of {} completed, LR = {} loss = {}'
                        .format(i_iter, cfg['trainer']['epochs']*len(data_loader), lr, losses.avg))

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # if rank == 0:
    #     logger.info('=========epoch[{}]=========,Train'.format(epoch))


def validate(model, data_loader, epoch):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']
    
    rank, world_size = get_rank(), get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if rank == 0:
        logger.info('=========epoch[{}]=========,Val'.format(epoch))

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        with torch.no_grad():
            preds = model(images,  is_eval=True)
           
        # get the output produced by model
        output = preds[0] if cfg['net'].get('aux_loss', False) else preds
        output = output.data.max(1)[1].cpu().numpy()
        target = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

        if step % 5 == 0 and rank==0:
            logger.info('iter = {} of {} completed'.format(step, len(data_loader)))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    accuracy_class = np.mean(accuracy_class)
    
    # print(mIoU)
    if rank == 0:
        logger.info('accuracy = {} mIoU = {}'.format(accuracy_class, mIoU))
    # torch.save(mIoU, 'eval_metric.pth.tar')
    return mIoU


if __name__ == '__main__':
    main()