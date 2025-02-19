import torch
import os, ast
NULL_ID_FOR_COREF = 0
def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters

def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold

def batch_extract_clusters(gold_clusters):
    gold_clusters = [extract_clusters(cl) for cl in gold_clusters]
    return gold_clusters

def batch_extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    return [extract_mentions_to_predicted_clusters_from_clusters(clusters) for clusters in gold_clusters]

#
def extract_clusters_for_decode(mention_to_antecedent):
    mention_to_antecedent = sorted(mention_to_antecedent)
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            clusters[cluster_idx].append(mention)
            mention_to_cluster[mention] = cluster_idx

        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster


def packed_data(sample_data, max_clusters, max_items_in_cluster, return_tensor=True, padding_value=0):
    if padding_value == 0:
        padding_item = [[0,0]]
    else:
        padding_item = [[-1,-1]]
    sample_data = ast.literal_eval(sample_data)
    empty_cluster = padding_item*max_items_in_cluster

    x_padded = [row + padding_item * (max_items_in_cluster - len(row)) for row in sample_data]
    
    number_padded_cluster = max_clusters - len(sample_data)
    x_padded =  x_padded + [empty_cluster]*number_padded_cluster
    if return_tensor:

        return torch.tensor(x_padded)
    return x_padded

def packed_list_data(list_sample_data, max_clusters, max_items_in_cluster, return_tensor=True, padding_value=0):
    list_output = []
    for i in list_sample_data:
        list_output.append(packed_data(i, max_clusters, max_items_in_cluster, 
                            return_tensor=True, padding_value=padding_value)
                          )
    if return_tensor:
        return torch.tensor(list_output)
    return list_output


def _range_item(item):
    if item[0] == item[1]:
        return [item[0]]
    return list(range(item[0], item[1] + 1))


def _range_cluster(cluster):
    result = []
    [result.extend(_range_item(item)) for item in cluster]
    result = sorted(list(set(result)))
    if result[0] == -1:
        return result[1:]
    return result

def single_mask(length, clusters):
    mask = [-1] * length
    group_ids = [_range_cluster(cluster) for cluster in clusters]
    for idx, item in enumerate(group_ids):
        if len(item) == 0:
            break
        min_group = min(item)
        for _item in item:
            mask[_item] = min_group
    return mask


def gen_masks(length, group_clusters):
    return [single_mask(length, clusters) for clusters in group_clusters]

def gen_cluster_embedding(pe, group_clusters):
    """Input:  position_embeddings: batch x length x dim
            clusters: batch x length
    Output: cluster embedding with same value at same cluster

    Example: tensor([[  [ 1,  1,  1,  1],
                        [ 2,  2,  2,  2],
                        [ 3,  3,  3,  3],
                        [ 4,  4,  4,  4],
                        [ 5,  5,  5,  5],
                        [ 6,  6,  6,  6],
                        [ 7,  7,  7,  7],
                        [ 8,  8,  8,  8],
                        [ 9,  9,  9,  9],
                        [10, 10, 10, 10]
                    ]])
             Group clusters: [[1, 0, 1, 1, 1, 1, 0, 2, 0, 2]]
                             [[0, -1, 0, 0, 0, 0, -1, 7, -1, 7]]
        =>  tensor([[   [ 1,  1,  1,  1],
                        [ 2,  2,  2,  2],
                        [ 1,  1,  1,  1],
                        [ 1,  1,  1,  1],
                        [ 1,  1,  1,  1],
                        [ 1,  1,  1,  1],
                        [ 7,  7,  7,  7],
                        [ 8,  8,  8,  8],
                        [ 9,  9,  9,  9],
                        [ 8,  8,  8,  8]
                    ]])

    """
    _mask = gen_masks(int(pe.shape[1]), group_clusters)
    mask = torch.tensor(_mask, dtype=float, device=pe.device)
    mask = mask.unsqueeze(dim=-1)

    mask_int = torch.tensor(_mask, dtype=torch.int64)
    ug_value = torch.where(mask == -1.0, pe, torch.tensor(0, dtype=pe.dtype, device=pe.device))


    g_value = [
        torch.where(
            mask[idx] == -1.0, torch.tensor(0, dtype=pe.dtype, device=pe.device), item[mask_int[idx]]
        )
        for idx, item in enumerate(pe)
    ]
    g_value = torch.stack(g_value)

    return g_value + ug_value