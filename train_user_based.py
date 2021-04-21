from data_loader.rating_loader import data_processing, load_train_data, load_tr_te_data
from model.model import MultiVAE, loss_function
from model.metric import ndcg_binary_at_k_batch, recall_at_k_batch
from utils.utils import plot_results, naive_sparse2tensor

import numpy as np
import random
import argparse
import json
import os
import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Train movie recommendation system on MovieLens dataset")
parser.add_argument('--config_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(data, index, batch_size, total_anneal_steps, anneal_cap, model, criterion, optimizer, device):
    model.train()
    losses = 0.0
    num_data = data.shape[0]
    num_batch = num_data // batch_size + 1
    np.random.shuffle(index)
    global update_count

    for batch_idx, start_idx in enumerate(range(0, num_data, batch_size)):
        end_idx = min(start_idx + batch_size, num_data)
        batch = data[index[start_idx:end_idx]]
        batch = naive_sparse2tensor(batch).to(device)

        if total_anneal_steps > 0:
            anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
        else:
            anneal = anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)

        loss = criterion(recon_batch, batch, mu, logvar, anneal)
        loss.backward()
        losses += loss.item()
        optimizer.step()
        update_count += 1

    return losses/num_batch


def evaluate(data_tr, data_te, index, batch_size, model, criterion, device):
    model.eval()
    losses = 0.0
    num_data = data_tr.shape[0]
    num_batch = num_data // batch_size + 1
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for start_idx in range(0, num_data, batch_size):
            end_idx = min(start_idx + batch_size, num_data)
            vad_tr = data_tr[index[start_idx:end_idx]]
            vad_te = data_te[index[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(vad_tr).to(device)

            recon_batch, mu, logvar = model(data_tensor)

            loss = criterion(recon_batch, data_tensor, mu, logvar)
            losses += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[vad_tr.nonzero()] = -np.inf

            n100 = ndcg_binary_at_k_batch(recon_batch, vad_te, 100)
            r20 = recall_at_k_batch(recon_batch, vad_te, 20)
            r50 = recall_at_k_batch(recon_batch, vad_te, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return losses/num_batch, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    set_seed(config['seed'])

    processed_data_dir = config['processed_data_dir']
    pos_threshold = config['pos_threshold']
    min_user_count = config['min_user_count']
    min_movie_count = config['min_movie_count']
    sort = True if config['sort'] == 1 else False
    num_user_test = config['num_user_test']

    specify_dir = '{}_{}_{}_{}_{}'.format(pos_threshold, min_user_count, min_movie_count, sort, num_user_test)
    processed_data_dir = os.path.join(processed_data_dir, specify_dir)

    if not os.path.exists(processed_data_dir):
        data_processing(config['rating_dir'],
                        config['movie_dir'],
                        processed_data_dir,
                        pos_threshold,
                        min_user_count,
                        min_movie_count,
                        num_user_test,
                        sort)

    unique_sid = list()
    with open(os.path.join(processed_data_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    train_data = load_train_data(os.path.join(processed_data_dir, 'train.csv'), n_items)
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(processed_data_dir, 'validation_tr.csv'),
                                               os.path.join(processed_data_dir, 'validation_te.csv'),
                                               n_items)

    save_dir = config['save_dir']
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
    writer.add_text('training_config', json.dumps(config))
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    device = config['device']
    p_dims = config['p_dims'] + [n_items]
    model = MultiVAE(p_dims)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_reg'])
    criterion = loss_function

    logging.info('Begin training')
    logging.info(config['msg'])
    print('Begin training')

    num_train = train_data.shape[0]
    num_val = vad_data_tr.shape[0]
    logging.info('Training size: {} | Validation size: {}'.format(num_train, num_val))
    print('Training size: {} | Validation size: {}'.format(num_train, num_val))

    epochs = config['epochs']
    tr_batch_size = config['tr_batch_size']
    te_batch_size = config['te_batch_size']
    train_index_list = list(range(num_train))
    vad_index_list = list(range(num_val))
    total_anneal_steps = config['total_anneal_steps']
    anneal_cap = config['anneal_cap']
    best_ndcg = -np.inf
    update_count = 0
    try:
        tr_losses = []
        val_losses = []
        val_ndcg_100 = []
        val_recall_20 = []
        val_recall_50 = []
        for epoch in range(1, epochs + 1):
            start = time.time()
            tr_loss = train(train_data, train_index_list, tr_batch_size, total_anneal_steps,
                            anneal_cap, model, criterion, optimizer, device)
            val_loss, ndcg_100, recall_20, recall_50 = evaluate(vad_data_tr, vad_data_te, vad_index_list,
                                                                te_batch_size, model, criterion, device)

            writer.add_scalars('losses', {'training': tr_loss, 'validation': val_loss}, epoch)
            writer.add_scalars('recall', {'20': recall_20, '50': recall_50}, epoch)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            val_ndcg_100.append(ndcg_100)
            val_recall_20.append(recall_20)
            val_recall_50.append(recall_50)

            if ndcg_100 > best_ndcg:
                with open(os.path.join(save_model_dir, "model.pt"), 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_ndcg = ndcg_100

            logging.info('| epoch {:3d} | time {:3.2f} s | tr_loss {:1.3f} | val_loss {:1.3f} | ndcg_100 {:1.3f} | '
                         'recall_20 {:1.3f} | recall_50 {:1.3f} | best_ndcg {:1.3f} |'
                         .format(epoch, (time.time() - start), tr_loss,
                                 val_loss, ndcg_100, recall_20, recall_50, best_ndcg))
            print('| epoch {:3d} | time {:3.2f} s | tr_loss {:1.3f} | val_loss {:1.3f} | ndcg_100 {:1.3f} | '
                  'recall_20 {:1.3f} | recall_50 {:1.3f} | best_ndcg {:1.3f} |'
                  .format(epoch, (time.time() - start), tr_loss, val_loss, ndcg_100, recall_20, recall_50, best_ndcg))

        plot_results(log_dir, 'loss', tr_losses, val_losses)
        writer.close()
    except KeyboardInterrupt:
        writer.close()
        print('Exiting from training early')
