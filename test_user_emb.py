from data_loader.rating_loader import load_tr_te_data
from data_loader.user_emb_loader import UserEmbeddingDataset
from model.batch import UserCollateFn
from model.model import UserEmbedding
from model.config import UserEmbeddingConfig
from model.metric import ndcg_binary_at_k_batch, recall_at_k_batch

import numpy as np
import random
import argparse
import json
import os
from scipy import sparse
import logging
import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="Test recommendation system model")
parser.add_argument('--save_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    save_path = args.save_path

    config = UserEmbeddingConfig(json.load(open(os.path.join(save_path, "config/config.json"))))
    logging.basicConfig(filename=os.path.join(save_path, 'log/test.log'), level=logging.INFO)

    te_data_tr, te_data_te = load_tr_te_data(os.path.join(config.data_dir, 'test_tr.csv'),
                                             os.path.join(config.data_dir, 'test_te.csv'),
                                             config.num_movie)

    te_data_tr = np.split(te_data_tr.indices, te_data_tr.indptr)[1:-1]
    te_data_te = np.split(te_data_te.indices, te_data_te.indptr)[1:-1]

    te_data_loader = DataLoader(
        UserEmbeddingDataset(te_data_tr, te_data_te),
        batch_size=config.vad_batch_size,
        shuffle=False,
        collate_fn=UserCollateFn(config))

    device = config.device
    model = UserEmbedding(config)
    model.load_state_dict(torch.load(os.path.join(save_path, "model/model.pt")))
    model.to(device)
    criterion = torch.nn.BCELoss()

    set_seed(config.seed)
    num_test = len(te_data_tr)

    print('Begin testing')
    logging.info('Begin testing')
    print('Testing size: {}'.format(num_test))
    logging.info('Testing size: {}'.format(num_test))
    try:
        te_loss, ndcg_100, r20, r50 = evaluate(te_data_loader, model, criterion, device)
        print('| val_loss {:1.3f} | ndcg_100 {:1.3f} | recall_20 {:1.3f} | recall_50 {:1.3f} |'
              .format(te_loss, ndcg_100, r20, r50))
        logging.info('| val_loss {:1.3f} | ndcg_100 {:1.3f} | recall_20 {:1.3f} | recall_50 {:1.3f} |'
                     .format(te_loss, ndcg_100, r20, r50))
    except KeyboardInterrupt:
        print('Exiting from training early')
