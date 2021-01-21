import os
import sys
import math
import time
import shutil
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

from thornet.logging.logger import WandbLogger
from thornet.config import Config
from thornet.utils.seed import set_seed
from thornet.models.util import count_params
from thornet.logging.meters import AverageMeter

from model import CytoSetModel
from data import CytoDatasetFromFCS
from utils import (
    EarlyStopping, load_fcs_dataset, train_valid_split, combine_samples
)


def test_valid(test_loader, model, args):
    """ Test the model performance """
    model.eval()
    losses = AverageMeter(round=3)
    correct_num, total_num = 0, 0
    y_pred, y_true = [], []

    for x, y in test_loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            prob = model(x)
            loss = F.binary_cross_entropy(prob, y, reduction='mean')
            pred_label = torch.ge(prob, 0.5)
        losses.update(loss.item(), n=x.size(0))
        v = (pred_label == y).sum()

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(prob.detach().cpu().numpy())

        correct_num += v.item()
        total_num += x.size(0)

    acc = float(correct_num) / total_num

    y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    return acc, losses.avg, auc


def test_model(test_samples, test_phenotypes, model, device):
    model.eval()
    correct_num, total_num = 0, 0
    y_pred, y_true = [], []
    losses = []

    for sample, label in zip(test_samples, test_phenotypes):
        with torch.no_grad():
            sample = torch.from_numpy(sample).to(device)
            true_label = torch.tensor([label], dtype=torch.float32).to(device)
            prob = model(sample)
            loss = F.binary_cross_entropy(prob, true_label, reduction='mean')
            pred_label = torch.ge(prob, 0.5)
        
        losses.append(loss.item())
        v = (pred_label == label).sum()

        y_true.append(label)
        y_pred.append(prob.detach().cpu().numpy())

        correct_num += v.item()
        total_num += 1
    
    acc = float(correct_num) / total_num
    
    y_true, y_pred = np.array(y_true), np.hstack(y_pred)
    auc = roc_auc_score(y_true, y_pred)
    eval_loss = np.mean(np.array(losses))

    return eval_loss, acc, auc


def train(args):
    set_seed(args.seed)

    logger = WandbLogger(
        logger_name=f'CytoSet-{args.ncell}@{args.pool}',
        log_dir=args.log_dir,
        stream=sys.stdout,
        args=args,
        wandb_project='CytoSet'
    )

    # set model
    model = CytoSetModel(args).to(args.device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.wts_decay
    )

    if args.ckpt is not None:
        print(f'Loading model from {args.ckpt}')
        checkpoint = torch.load(args.ckpt, map_location='cpu' if not torch.cuda.is_available() else None)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])

    # set dataloader
    train_samples, train_phenotypes = load_fcs_dataset(
        args.train_fcs_info, args.markerfile, args.co_factor
    )
    test_samples, test_phenotypes = load_fcs_dataset(
        args.test_fcs_info, args.markerfile, args.co_factor
    )
    valid_phenotypes = train_phenotypes
    
    if (args.valid_fcs_info is not None) or args.generate_valid:
        if args.valid_fcs_info is not None:
            valid_samples, valid_phenotypes = load_fcs_dataset(
                args.valid_fcs_info, args.marker_file, args.co_factor
            )
            X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
            X_valid, id_valid = combine_samples(valid_samples, np.arange(len(valid_samples)))
        else:
            X_train, id_train, X_valid, id_valid = train_valid_split(
                train_samples, np.arange(len(train_samples))
            )
        X_train, id_train = shuffle(X_train, id_train)    
        train_data = CytoDatasetFromFCS(X_train, id_train, train_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)
        valid_data = CytoDatasetFromFCS(X_valid, id_valid, valid_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)
    else:
        X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
        X_train, id_train = shuffle(X_train, id_train)
        train_data = CytoDatasetFromFCS(X_train, id_train, train_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)
        logger.info("Neither having valid dataset nor generating valid dataset, use train data as valid dataset")
        valid_data = CytoDatasetFromFCS(X_train, id_train, train_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=1,
        drop_last=True,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    logger.info('**** Start Training ****')
    logger.info(f' config: {args.ncell}@{args.pool}')
    logger.info(f' Total epochs: {args.n_epochs}')
    logger.info('Total Params: {:.2f}M'.format(count_params(model) / 1e6))

    losses = AverageMeter(round=3)
    data_time = AverageMeter(round=3)
    step_time = AverageMeter(round=3)

    best_auc = 0
    pbar = tqdm(range(args.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)
    
    # start the main training loop
    for epoch in pbar:
        model.train()

        # get the data
        for x, y in train_loader:
            start_time = time.time()
            x, y = x.to(args.device), y.to(args.device)
            # count data moving time
            data_time.update(time.time() - start_time)

            # model feed forward
            prob = model(x)
            loss = F.binary_cross_entropy(prob, y, reduction='mean')

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_time.update(time.time() - start_time)
            losses.update(loss.item())
        
        # log the training progress
        if (epoch + 1) % args.log_interval == 0:
            val_acc, val_loss, val_auc = test_valid(valid_loader, model, args)

            pbar.set_description(
                "Epoch: {}/{}, data: {:.3f}, step: {:.3f}, loss: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}, val_auc: {:.3f}".format(
                    str(epoch + 1).zfill(4), args.n_epochs, data_time.avg,
                    step_time.avg, losses.avg, val_loss, val_acc, val_auc
                )
            )

            stats = {
                'epoch': epoch + 1,
                'loss': losses.avg,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc
            }
            logger._log_to_wandb(stats=stats, epoch=epoch + 1)

            # check early stop condition
            early_stopping(val_loss=val_loss)
            if early_stopping.early_stop:
                logger.info(f"Training early stops at epoch: {epoch+1}")
                break

        losses.reset()
        data_time.reset()
        step_time.reset()

        if (epoch + 1) % args.save_interval == 0:
            val_acc, val_loss, val_auc = test_valid(valid_loader, model, args)
            is_best = val_auc >= best_auc

            ckpt_file = f"{args.log_dir}/{str(epoch + 1).zfill(4)}.ckpt"

            torch.save(
                {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': args,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'epoch': epoch + 1
                },
                ckpt_file
            )
            if is_best:
                # save the best model and check points
                torch.save(model.state_dict(), f'{args.log_dir}/best_model.pt')
                shutil.copyfile(ckpt_file, f'{args.log_dir}/best.ckpt')

        pbar.update()

    pbar.close()
    logger.info("**** Training Finished ****")

    # load best model and check the performance on test data
    # format test samples
    test_samples = [np.expand_dims(sample, 0).astype(np.float32) for sample in test_samples]
    state_dict = torch.load(f'{args.log_dir}/best_model.pt', map_location='cpu' if not torch.cuda.is_available else None)
    model.load_state_dict(state_dict, strict=True)
    _, test_acc, test_auc = test_model(test_samples, test_phenotypes, model, args.device)
    logger.info("Testing Acc: {:.3f}, Testing Auc: {:.3f}".format(test_acc, test_auc))
    
    # Finished the training and testing, saving the configurations
    logger.info("Testing finished, saving training configurations....")
    config = Config.from_args(args)
    config.to_json_file(f"{args.log_dir}/config.json")
    logger.info("Done")


def main():
    parser = argparse.ArgumentParser("Cytometry Set Model")

    # model
    parser.add_argument('--in_dim', default=37, type=int, help="input dim")
    parser.add_argument('--h_dim', default=64, type=int, help='hidden dims to use in the model')
    parser.add_argument('--pool', default='max', choices=['mean', 'max', 'sum'], type=str, help='block pooling type')
    parser.add_argument('--out_pool', default='mean', choices=['mean', 'max', 'sum'], type=str, help='output pooling type')
    parser.add_argument('--nblock', default=1, type=int, help="# of blocks to use in the model")

    # optimizer
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta_1 params in the optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta_2 params in the optimizer')
    parser.add_argument('--wts_decay', default=1e-3, type=float, help='coefficient of weight decay')
    parser.add_argument('--patience', default=5, type=int, help='the patience param for early stopping')

    # data
    parser.add_argument('--train_fcs_info', type=str, help='path to train fcs info file')
    parser.add_argument('--valid_fcs_info', default=None, type=str, help='path to valid fcs info file')
    parser.add_argument('--test_fcs_info', type=str, help='path to test fcs info file')
    parser.add_argument('--markerfile', type=str, help='path to marker indication file')
    parser.add_argument('--generate_valid', action='store_true', help='whether to generate valid data from train data')
    
    parser.add_argument('--batch_size', default=200, type=int, help='batch size of labeled data')
    parser.add_argument('--nsubset', default=1024, type=int, help='total number of multi-cell inputs that will be generated per class')
    parser.add_argument('--ncell', default=200, type=int, help='number of cells per multi-cell input')
    parser.add_argument('--co_factor', default=5, type=float, help='arcsinh normalization factor')
    parser.add_argument('--per_sample', action='store_true', help='whether the nsubset argument refers to each class or each input')

    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the data')
    parser.add_argument('--n_epochs', default=200, type=int, help='number of total training epochs')
    parser.add_argument('--log_dir', default='./exp', type=str, help='path to log dir')
    parser.add_argument('--log_interval', default=1, type=int, help='logging interval')
    parser.add_argument('--save_interval', default=5, type=int, help='save model interval')

    # utils
    parser.add_argument('--seed', default=12345, type=int, help='random seed to use')
    parser.add_argument('--device', default='cuda', type=str, help='specify the training device')
    parser.add_argument('--ckpt', default=None, type=str, help='path to the checkpoint file')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    # train the model
    train(args)


if __name__ == "__main__":
    main()
