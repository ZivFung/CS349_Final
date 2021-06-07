
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import DataLoader
import torch.utils.data as data
import utils
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models import sentimentAnalyze1
import utils
from json import dumps
import random
from args import get_train_args

def get_model(log,args,word_vectors):
    model=sentimentAnalyze1(word_vectors,100,256,2,0.1)

    model=nn.DataParallel(model,args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = utils.load_model(model, args.load_path, args.gpu_ids)

    else:
        step = 0

    return model,step

def main(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = utils.torch_from_json(args.word_emb_file)
    #char_vectors = utils.torch_from_json(args.char_emb_file)
    # Get GNN model
    log.info('Building model...')
    model,step=get_model(log,args,word_vectors)
    model.to(device)
    model.train()

    ema = utils.EMA(model, args.ema_decay)

    # Get saver
    saver = utils.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)


    optimizer = optim.Adam(model.parameters(), args.lr,
                            weight_decay=args.l2_wd)
    scheduler=sched.LambdaLR(optimizer,lambda step: 1)

    # Get data loader
    log.info('Building dataset...')
    train_dataset = utils.TwitterDataset(args.train_path)
    val_dataset =utils.TwitterDataset(args.val_path)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=utils.train_collate_fn)

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=utils.train_collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs,y in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                batch_size = cw_idxs.size(0)
                y=y.to(device)
                optimizer.zero_grad()

                # Forward
                pred= model(cw_idxs)
                weight_vector=torch.ones([2],device=device)
                num_positive_sample=torch.sum(y.float())
                if num_positive_sample>0:
                    weight=batch_size-torch.sum(y.float())/torch.sum(y.float())
                    weight_vector[1]=weight
                loss = F.nll_loss(pred, y,weight=weight_vector)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results = evaluate(model, val_loader, device)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)


def evaluate(model,  data_loader, device):
    nll_meter = utils.AverageMeter()
    metric_meter=utils.MetricsMeter()
    model.eval()
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:

        for cw_idxs, cc_idxs,y in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            y=y.to(device)
            batch_size = cw_idxs.size(0)
            # Forward
            pred= model(cw_idxs)
            weight_vector=torch.ones([2],device=device)
            num_positive_sample=torch.sum(y.float())
            if num_positive_sample>0:
                weight=batch_size-torch.sum(y.float())/torch.sum(y.float())
                weight_vector[1]=weight
            loss = F.nll_loss(pred, y,weight=weight_vector)
            nll_meter.update(loss.item(), batch_size)
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            pred=pred.exp()[:,1].detach().cpu()
            metric_meter.update(pred,y.cpu().int())

    model.train()

    metrics_result = metric_meter.return_metrics()

    results_list = [
        ('Loss', nll_meter.avg),
        ('Accuracy', metrics_result["Accuracy"]),
        ('Recall', metrics_result["Recall"]),
        ('Precision', metrics_result["Precision"]),
        ('Specificity', metrics_result["Specificity"]),
        ('F1', metrics_result["F1"]),
        ("AUC", metrics_result["AUC"])
    ]
    results = OrderedDict(results_list)

    return results


if __name__=="__main__":
    main(get_train_args())