from data_loader.rating_loader import load_tr_te_data
from model.model import MultiVAE, loss_function
from model.metric import ndcg_binary_at_k_batch, recall_at_k_batch
from utils.utils import naive_sparse2tensor

import numpy as np
import random
import argparse
import json
import os
import logging
import torch


parser = argparse.ArgumentParser(description="Test recommendation system model")
parser.add_argument('--save_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    save_path = args.save_path

    config = json.load(open(os.path.join(save_path, "config/config.json")))
    logging.basicConfig(filename=os.path.join(save_path, 'log/test.log'), level=logging.INFO)

    processed_data_dir = config['processed_data_dir']
    specify_dir = '{}_{}_{}_{}_{}'.format(config['pos_threshold'],
                                          config['min_user_count'],
                                          config['min_movie_count'],
                                          True if config['sort'] == 1 else False,
                                          config['num_user_test'])

    processed_data_dir = os.path.join(processed_data_dir, specify_dir)

    unique_sid = list()
    with open(os.path.join(processed_data_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    test_data_tr, test_data_te = load_tr_te_data(os.path.join(processed_data_dir, 'test_tr.csv'),
                                                 os.path.join(processed_data_dir, 'test_te.csv'),
                                                 n_items)

    device = config['device']
    p_dims = config['p_dims'] + [n_items]
    model = MultiVAE(p_dims)
    model.load_state_dict(torch.load(os.path.join(save_path, "model/model.pt")))
    model.to(device)
    criterion = loss_function

    set_seed(config['seed'])
    num_test = test_data_tr.shape[0]

    print('Begin testing')
    logging.info('Begin testing')
    print('Testing size: {}'.format(num_test))
    logging.info('Testing size: {}'.format(num_test))
    try:
        val_loss, ndcg_100, recall_20, recall_50 = evaluate(test_data_tr, test_data_te, list(range(num_test)),
                                                            config['te_batch_size'], model, criterion, device)
        print('| val_loss {:1.3f} | ndcg_100 {:1.3f} | recall_20 {:1.3f} | recall_50 {:1.3f} |'
              .format(val_loss, ndcg_100, recall_20, recall_50))
        logging.info('| val_loss {:1.3f} | ndcg_100 {:1.3f} | recall_20 {:1.3f} | recall_50 {:1.3f} |'
                     .format(val_loss, ndcg_100, recall_20, recall_50))
    except KeyboardInterrupt:
        print('Exiting from training early')
