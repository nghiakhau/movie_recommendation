import sys
import os

import pandas as pd
import numpy as np
from scipy import sparse

from utils.utils import get_count_df


def filter_triplets(data, min_user_count=0, min_movie_count=0):
    """
        min_user_count: only keep the user who gives at least min_user_count ratings
        min_movie_count: only keep the movie that receives at least min_movie_count ratings
    """

    if min_movie_count > 0:
        movie_count = get_count_df(data, 'movieId')
        data = data[data['movieId'].isin(movie_count[movie_count['size'] >= min_movie_count]['movieId'])]

    # After doing this, some of the movies will have less than min_uc users, but should only be a small proportion
    if min_user_count > 0:
        user_count = get_count_df(data, 'userId')
        data = data[data['userId'].isin(user_count[user_count['size'] >= min_user_count]['userId'])]

    # Update both user_count and movie_count after filtering
    user_count, movie_count = get_count_df(data, 'userId'), get_count_df(data, 'movieId')
    return data, user_count, movie_count


def split_train_test_proportion(data, sorted_by_timestamp, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u < 2:
            continue

        idx = np.zeros(n_items_u, dtype='bool')
        test_size = max(1, int(test_prop * n_items_u))

        if sorted_by_timestamp:
            group = group.sort_values('timestamp')
            idx[-test_size:] = True
        else:
            idx[np.random.choice(n_items_u, size=test_size, replace=False).astype('int64')] = True

        tr_list.append(group[np.logical_not(idx)])
        te_list.append(group[idx])

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(data, userid2id, movieid2id):
    uid = data['userId'].apply(lambda x: userid2id[x])
    sid = data['movieId'].apply(lambda x: movieid2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


def load_train_data(csv_file, n_items):
    df = pd.read_csv(csv_file)
    n_users = df['uid'].max() + 1

    rows, cols = df['uid'], df['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    df_tr = pd.read_csv(csv_file_tr)
    df_te = pd.read_csv(csv_file_te)

    start_idx = min(df_tr['uid'].min(), df_te['uid'].min())
    end_idx = max(df_tr['uid'].max(), df_te['uid'].max())

    rows_tr, cols_tr = df_tr['uid'] - start_idx, df_tr['sid']
    rows_te, cols_te = df_te['uid'] - start_idx, df_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))

    # remove the rows whose all elements are zero
    tr_all_zeros = np.where(np.array(data_tr.getnnz(1) < 1) == True)[0]
    te_all_zeros = np.where(np.array(data_te.getnnz(1) < 1) == True)[0]
    if len(tr_all_zeros) > 0 or len(te_all_zeros) > 0:
        print('tr_data: detect rows {} whose all elements are zero. Removing...'.format(tr_all_zeros))
        print('te_data: detect rows {} whose all elements are zero. Removing...'.format(te_all_zeros))
        data_tr = data_tr[data_tr.getnnz(1) > 0]
        data_te = data_te[data_te.getnnz(1) > 0]

    return data_tr, data_te


def data_processing(rating_dir, movie_dir, processed_data_dir, pos_threshold,
                    min_user_count, min_movie_count, num_user_test, sort):
    """
        pos_threshold: the rating that we consider an user like ENOUGH that movie (4.0 ?)
        min_user_count: only keep the user who gives at least min_user_count ratings
        min_movie_count: only keep the movie that receives at least min_movie_count ratings
        sort: sort the ratings on timestamp before splitting data
    """
    os.makedirs(processed_data_dir)

    print("Load and process MovieLens dataset")
    ratings = pd.read_csv(rating_dir)
    movies = pd.read_csv(movie_dir)  # movie with available description and genre
    ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
    ratings = ratings[ratings['rating'] >= pos_threshold]

    n_user = len(ratings['userId'].unique())
    n_movie = len(ratings['movieId'].unique())
    sparsity = 1. * ratings.shape[0] / (n_user * n_movie)
    with open(os.path.join(processed_data_dir, 'processing_stat.txt'), 'w') as f:
        f.write("Before filtering, there are %d ratings from %d users and %d movies (sparsity: %.3f%%)" %
                (ratings.shape[0], n_user, n_movie, sparsity * 100))
        f.write("\n")
    print("Before filtering, there are %d ratings from %d users and %d movies (sparsity: %.3f%%)" %
          (ratings.shape[0], n_user, n_movie, sparsity * 100))

    ratings, user_activity, movie_popularity = filter_triplets(ratings, min_user_count, min_movie_count)

    sparsity = 1. * ratings.shape[0] / (user_activity.shape[0] * movie_popularity.shape[0])
    with open(os.path.join(processed_data_dir, 'processing_stat.txt'), 'a') as f:
        f.write("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
                (ratings.shape[0], user_activity.shape[0], movie_popularity.shape[0], sparsity * 100))
        f.write("\n")
    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (ratings.shape[0], user_activity.shape[0], movie_popularity.shape[0], sparsity * 100))

    unique_uid = user_activity['userId'].to_numpy()
    n_users = len(unique_uid)
    idx_perm = np.random.permutation(n_users)
    unique_uid = unique_uid[idx_perm]

    tr_users = unique_uid[:(n_users - num_user_test * 2)]
    vd_users = unique_uid[(n_users - num_user_test * 2): (n_users - num_user_test)]
    te_users = unique_uid[(n_users - num_user_test):]

    train_ratings = ratings.loc[ratings['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_ratings['movieId'])
    with open(os.path.join(processed_data_dir, 'processing_stat.txt'), 'a') as f:
        f.write('We only recommend the movie that exists in training data. There are {} movies'.format(len(unique_sid)))
        f.write("\n")
    print('We only recommend the movie that exists in training data. There are {} movies'.format(len(unique_sid)))

    movieid2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    userid2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    with open(os.path.join(processed_data_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_ratings = ratings.loc[ratings['userId'].isin(vd_users)]
    vad_ratings = vad_ratings.loc[vad_ratings['movieId'].isin(unique_sid)]
    vad_ratings_tr, vad_ratings_te = split_train_test_proportion(vad_ratings, sort, test_prop=0.2)

    test_ratings = ratings.loc[ratings['userId'].isin(te_users)]
    test_ratings = test_ratings.loc[test_ratings['movieId'].isin(unique_sid)]
    test_ratings_tr, test_ratings_te = split_train_test_proportion(test_ratings, sort, test_prop=0.2)

    train_data = numerize(train_ratings, userid2id, movieid2id)
    train_data.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_ratings_tr, userid2id, movieid2id)
    vad_data_tr.to_csv(os.path.join(processed_data_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_ratings_te, userid2id, movieid2id)
    vad_data_te.to_csv(os.path.join(processed_data_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_ratings_tr, userid2id, movieid2id)
    test_data_tr.to_csv(os.path.join(processed_data_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_ratings_te, userid2id, movieid2id)
    test_data_te.to_csv(os.path.join(processed_data_dir, 'test_te.csv'), index=False)

