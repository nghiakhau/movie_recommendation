from data_loader.movie_loader import MovieDataset
from model.model import MovieEmbedding
from model.batch import TokenizersCollateFn
from model.metric import compute_accuracy
from utils.utils import plot_results

import numpy as np
import pandas as pd
import random
import argparse
import json
import os
import logging
import time
import ast
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Train movie embedding on MovieLens dataset")
parser.add_argument('--config_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(data, model, criterion, optimizer, accumulation_steps, scheduler, device):
    model.train()
    losses = 0.0
    accuracy = []

    model.zero_grad()
    for step, batch in enumerate(data):
        batch.to_device(device)

        y_prob = model(batch.text, batch.attention_mask)

        loss = criterion(y_prob, batch.label)
        losses += loss.item()
        acc = compute_accuracy(y_prob, batch.label)
        accuracy.append(acc.cpu())

        loss /= accumulation_steps          # Normalize our loss (if averaged)
        loss.backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return losses/len(data), np.mean(accuracy)


def evaluate(data, model, criterion, device):
    model.eval()
    losses = 0.0
    accuracy = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            batch.to_device(device)

            y_prob = model(batch.text, batch.attention_mask)

            loss = criterion(y_prob, batch.label)
            losses += loss.item()
            acc = compute_accuracy(y_prob, batch.label)
            accuracy.append(acc.cpu())

    return losses/len(data), np.mean(accuracy)


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    set_seed(config['seed'])

    movies = pd.read_csv(config['data_dir'])
    movies.drop_duplicates('movieId', inplace=True)
    movies.rename(columns={'overview': 'text'}, inplace=True)
    movies['genres'] = movies['genres'].map(ast.literal_eval)
    movies['keywords'] = movies['keywords'].map(ast.literal_eval)

    unique_sid = list()
    with open(config['selected_id_dir'], 'r') as f:
        for line in f:
            unique_sid.append(int(line.strip()))

    movies = movies[movies['movieId'].isin(unique_sid)]

    train_data, vad_data = train_test_split(movies, test_size=0.2)
    train_data = train_data.reset_index().drop('index', 1)
    vad_data = vad_data.reset_index().drop('index', 1)

    tr_data_loader = DataLoader(
                        MovieDataset(train_data),
                        batch_size=config['tr_batch_size'],
                        shuffle=True,
                        collate_fn=TokenizersCollateFn(config))
    vad_data_loader = DataLoader(
                        MovieDataset(vad_data),
                        batch_size=config['vad_batch_size'],
                        shuffle=False,
                        collate_fn=TokenizersCollateFn(config))

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

    model = MovieEmbedding(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_reg'])
    criterion = torch.nn.BCELoss()
    epochs = config['epochs']
    total_steps = len(tr_data_loader) // config['accumulate_grad_batches'] * epochs
    print('total_steps', total_steps)
    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=config['warmup_steps'],
                        num_training_steps=total_steps)

    logging.info('Begin training')
    logging.info(config['msg'])
    print('Begin training')

    num_train = train_data.shape[0]
    num_val = vad_data.shape[0]
    logging.info('Training size: {} | Validation size: {}'.format(num_train, num_val))
    print('Training size: {} | Validation size: {}'.format(num_train, num_val))

    best_acc = -np.inf
    freeze_encoder_after_epochs = config['freeze_encoder_after_epochs']
    accumulate_grad_batches = config['accumulate_grad_batches']
    try:
        tr_losses = []
        vad_losses = []
        tr_accuracy = []
        vad_accuracy = []
        for epoch in range(1, epochs + 1):

            if (epoch - 1) == freeze_encoder_after_epochs:
                logging.info('Start to freeze the encoder...')
                print('Start to freeze the encoder...')
                for name, param in model.encoder.named_parameters():
                    param.requires_grad = False
                # Maybe re-create data_loader with a bigger batch size ???

            start = time.time()
            tr_loss, tr_acc = train(tr_data_loader, model, criterion, optimizer,
                                    accumulate_grad_batches, scheduler, device)
            vad_loss, vad_acc = evaluate(vad_data_loader, model, criterion, device)

            writer.add_scalars('loss', {'training': tr_loss, 'validation': vad_loss}, epoch)
            writer.add_scalars('accuracy', {'training': tr_acc, 'validation': vad_acc}, epoch)
            tr_losses.append(tr_loss)
            vad_losses.append(vad_loss)
            tr_accuracy.append(tr_acc)
            vad_accuracy.append(vad_acc)

            if vad_acc > best_acc:
                with open(os.path.join(save_model_dir, "model.pt"), 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_acc = vad_acc

            logging.info('| epoch {:3d} | time {:3.2f} s | tr_loss {:1.3f} | '
                         'vad_loss {:1.3f} | tr_acc {:1.3f} |  vad_acc {:1.3f} |'
                         .format(epoch, (time.time() - start), tr_loss, vad_loss, tr_acc, vad_acc))
            print('| epoch {:3d} | time {:3.2f} s | tr_loss {:1.3f} | vad_loss {:1.3f}'
                  ' | tr_acc {:1.3f} | vad_acc {:1.3f} |'
                  .format(epoch, (time.time() - start), tr_loss, vad_loss, tr_acc, vad_acc))

        plot_results(log_dir, 'loss', tr_losses, vad_losses)
        writer.close()
    except KeyboardInterrupt:
        writer.close()
        print('Exiting from training early')
