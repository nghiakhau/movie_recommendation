from data_loader.rating_loader import load_train_data, load_tr_te_data
from data_loader.user_emb_loader import UserEmbeddingDataset
from model.config import UserEmbeddingConfig
from model.batch import UserCollateFn
from model.model import UserEmbedding
from model.metric import ndcg_binary_at_k_batch, recall_at_k_batch
from utils.utils import plot_results

from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import random
import argparse
import json
import os
import logging
import time
from scipy import sparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Train user embedding using movie embedding from MovieLens dataset")
parser.add_argument('--config_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(data, model, criterion, optimizer, accumulation_steps, scheduler, device):
    model.train()
    losses = 0.0

    model.zero_grad()
    for step, batch in enumerate(data):
        batch.to_device(device)

        y_prob = model(batch.tr_idx)

        loss = criterion(y_prob, batch.label)
        losses += loss.item()

        # loss /= accumulation_steps          # Normalize our loss (if averaged)
        loss.backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return losses/len(data)


def evaluate(data, model, criterion, device):
    model.eval()
    losses = 0.0
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            batch.to_device(device)

            y_prob = model(batch.tr_idx)

            loss = criterion(y_prob, batch.label)
            losses += loss.item()

            y_prob = y_prob.cpu()
            target = np.zeros_like(y_prob)
            for y in range(y_prob.size(0)):
                y_prob[y, batch.tr_idx_wo_pad[y]] = -np.inf
                target[y, batch.te_idx[y]] = 1.0
            y_prob = y_prob.numpy()
            target = sparse.csr_matrix(target)

            n100 = ndcg_binary_at_k_batch(y_prob, target, 100)
            re20 = recall_at_k_batch(y_prob, target, 20)
            re50 = recall_at_k_batch(y_prob, target, 50)

            n100_list.append(n100)
            r20_list.append(re20)
            r50_list.append(re50)

        n100_list = np.concatenate(n100_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

    return losses / len(data), np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


if __name__ == '__main__':
    args = parser.parse_args()
    config = UserEmbeddingConfig(json.load(open(args.config_path)))

    set_seed(config.seed)
    processed_data_dir = os.path.join(config.data_dir)

    train_data = load_train_data(os.path.join(processed_data_dir, 'train.csv'), config.num_movie)
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(processed_data_dir, 'validation_tr.csv'),
                                               os.path.join(processed_data_dir, 'validation_te.csv'),
                                               config.num_movie)

    train_data = np.split(train_data.indices, train_data.indptr)[1:-1]
    vad_data_tr = np.split(vad_data_tr.indices, vad_data_tr.indptr)[1:-1]
    vad_data_te = np.split(vad_data_te.indices, vad_data_te.indptr)[1:-1]

    tr_data_loader = DataLoader(
        UserEmbeddingDataset(train_data, train_data),
        batch_size=config.tr_batch_size,
        shuffle=True,
        collate_fn=UserCollateFn(config))
    vad_data_loader = DataLoader(
        UserEmbeddingDataset(vad_data_tr, vad_data_te),
        batch_size=config.vad_batch_size,
        shuffle=False,
        collate_fn=UserCollateFn(config))

    save_dir = config.save_dir
    save_model_dir = os.path.join(save_dir, 'model')
    log_dir = os.path.join(save_dir, 'log')
    config_dir = os.path.join(save_dir, 'config')

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
    writer = SummaryWriter(log_dir)
    writer.add_text('training_config', json.dumps(json.load(open(args.config_path))))
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(json.load(open(args.config_path)), f, indent=2)

    device = config.device
    model = UserEmbedding(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    criterion = torch.nn.BCELoss()
    epochs = config.epochs
    accumulate_grad_batches = config.accumulate_grad_batches
    total_steps = len(tr_data_loader) // accumulate_grad_batches * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps)

    logging.info('Begin training')
    logging.info('Training total steps: '.format(total_steps))
    logging.info(config.msg)
    print('Begin training')
    print('Training total steps', total_steps)

    num_train = len(train_data)
    num_val = len(vad_data_tr)
    logging.info('Training size: {} | Validation size: {}'.format(num_train, num_val))
    print('Training size: {} | Validation size: {}'.format(num_train, num_val))

    best_acc = -np.inf
    try:
        tr_losses = []
        vad_losses = []
        vad_ndcg_100 = []
        vad_recall_20 = []
        vad_recall_50 = []
        for epoch in range(1, epochs + 1):

            start = time.time()
            tr_loss = train(tr_data_loader, model, criterion, optimizer, accumulate_grad_batches, scheduler, device)
            vad_loss, ndcg_100, r20, r50 = evaluate(vad_data_loader, model, criterion, device)

            writer.add_scalars('loss', {'training': tr_loss, 'validation': vad_loss}, epoch)
            writer.add_scalars('recall', {'20': r20, '50': r50}, epoch)
            tr_losses.append(tr_loss)
            vad_losses.append(vad_loss)
            vad_ndcg_100.append(ndcg_100)
            vad_recall_20.append(r20)
            vad_recall_50.append(r50)

            if ndcg_100 > best_acc:
                with open(os.path.join(save_model_dir, "model.pt"), 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_acc = ndcg_100

            logging.info('| epoch {:3d} | time {:3.2f} s | tr_loss {:1.3f} | '
                         'vad_loss {:1.3f} | ndcg_100 {:1.3f} |  r20 {:1.3f} |  r50 {:1.3f} |'
                         .format(epoch, (time.time() - start), tr_loss, vad_loss, ndcg_100, r20, r50))
            print('| epoch {:3d} | time {:3.2f} s | tr_loss {:1.3f} |  vad_loss {:1.3f} | '
                  'ndcg_100 {:1.3f} |  r20 {:1.3f} |  r50 {:1.3f} |'
                  .format(epoch, (time.time() - start), tr_loss, vad_loss, ndcg_100, r20, r50))

        plot_results(log_dir, 'loss', tr_losses, vad_losses)
        plot_results(log_dir, 'recall', vad_recall_20, vad_recall_50)
        writer.close()
    except KeyboardInterrupt:
        writer.close()
        print('Exiting from training early')
