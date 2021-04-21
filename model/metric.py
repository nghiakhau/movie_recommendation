import numpy as np


def ndcg_binary_at_k_batch(x_pred, k_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    
    batch_users = x_pred.shape[0]
    idx_topk_part = np.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (k_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in k_batch.getnnz(axis=1)])
    return dcg / idcg


def recall_at_k_batch(x_pred, k_batch, k=100):
    batch_users = x_pred.shape[0]

    idx = np.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (k_batch > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))

    return recall


def compute_accuracy(pred, target, threshold=0.5):
    binary_pred = (pred > threshold).float()
    binary_pred += target

    return ((binary_pred == 2).sum() + (binary_pred == 0).sum()).float() / (pred.shape[0] * pred.shape[1])

