import torch
import os
import matplotlib.pyplot as plt


def get_count_df(data, column):
    grouped = data[[column]].groupby(column, as_index=False)
    count = grouped.size()
    return count


def plot_results(save_dir, name, train_results, val_results):
    plt.style.use('ggplot')
    plt.plot(train_results, label='training ' + name)
    plt.plot(val_results, label='validation ' + name)
    plt.legend()
    plt.savefig(os.path.join(save_dir, name + '.jpg'))
    plt.close()


def naive_sparse2tensor(x):
    return torch.FloatTensor(x.toarray())


def genre2label(genres, genre2id, num_genre):
    result = [0.0] * num_genre
    for g in genres:
        result[genre2id[g]] = 1.0

    return result






