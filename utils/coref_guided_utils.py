import sys, os
import logging
import random
import torch
import torch.nn.functional as F
import numpy as np
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
    # gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    # gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
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

# padding_item = [[0,0]] # need to change?

def packed_data(sample_data, max_clusters, max_items_in_cluster, return_tensor=True, padding_value=0):
    if padding_value == 0:
        padding_item = [[0,0]]
    else:
        padding_item = [[-1,-1]]
    # max_cluster x max_item x 2
    sample_data = ast.literal_eval(sample_data)
    empty_cluster = padding_item*max_items_in_cluster

    # fill current list to match max_item
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

"""
For coref-guided probabilities
"""

def _build_span_indices(starts_ends_input, padded):
    starts, ends = starts_ends_input[:, 0], starts_ends_input[:, 1]

    # Create a matrix where each row is a sequence of numbers starting from the corresponding start index
    rows = torch.arange(0, torch.max(ends-starts) + 1).unsqueeze(0) + starts.unsqueeze(1)

    # Clip the numbers so that numbers beyond the "end" index are set to the "end" index
    clipped_rows = torch.min(rows, ends.unsqueeze(1))

    return clipped_rows

def build_casterian_indices(doc_indices, mention_to_antecedent, mention_indices, antecedent_indices ):
    assert doc_indices.shape[0] == mention_to_antecedent.shape[0]
    mention_idx_id1 = mention_to_antecedent[:,0,:]
    mention_idx_id2 = mention_to_antecedent[:,1,:]
    # max_pad_id1 = mention_idx_id1[:, 1] - mention_idx_id1[:, 0] + 1
    # max_pad_id2 = mention_idx_id2[:, 1] - mention_idx_id2[:, 0] + 1
    max_pad_id1 = mention_idx_id1[:, 1] - mention_idx_id1[:, 0] 
    max_pad_id2 = mention_idx_id2[:, 1] - mention_idx_id2[:, 0]     

    span_idx_id1 = _build_span_indices(mention_idx_id1, max_pad_id1)
    span_idx_id2 = _build_span_indices(mention_idx_id2, max_pad_id2)

    # Get Cartesian product indices for span_idx_id1 and span_idx_id2
    span_idx_id1_cartesian = span_idx_id1.unsqueeze(2).repeat(1, 1, span_idx_id2.size(1))
    span_idx_id2_cartesian = span_idx_id2.unsqueeze(1).repeat(1, span_idx_id1.size(1), 1)

    # Flatten the tensors to have one dimension with all Cartesian product combinations
    span_idx_id1_flat = span_idx_id1_cartesian.reshape(span_idx_id1_cartesian.size(0), -1)
    span_idx_id2_flat = span_idx_id2_cartesian.reshape(span_idx_id2_cartesian.size(0), -1)

    # Expand X for the Cartesian product size
    doc_idx_exp = doc_indices[:, None].expand(-1, span_idx_id1_flat.size(1)).reshape(-1)
    mention_indices_exp = mention_indices[:, None].expand(-1, span_idx_id1_flat.size(1)).reshape(-1)
    antecedent_indices_exp = antecedent_indices[:, None].expand(-1, span_idx_id1_flat.size(1)).reshape(-1)

    # Flatten Y and Z for indexing
    Y_indices = span_idx_id1_flat.reshape(-1)
    Z_indices = span_idx_id2_flat.reshape(-1)

    return doc_idx_exp, Y_indices, Z_indices, mention_indices_exp, antecedent_indices_exp

def build_prob_and_mask_matrices(max_length, batch_num, initialized_type=0):
    # Create a tensor of ones
    if initialized_type == 1:
        _mask = torch.ones(max_length, max_length)
    else:
        _mask = torch.zeros(max_length, max_length)

    # create probs
    probs = _mask.unsqueeze(0).expand(batch_num, -1, -1)

    # Fill the upper diagonal portion with 0
    _mask_upper_indices = torch.triu_indices(max_length, max_length, offset=1)
    _mask[_mask_upper_indices[0], _mask_upper_indices[1]] = 0
    # Create a tensor of shape (M, N, N) by repeating the matrix M times
    masks = _mask.unsqueeze(0).expand(batch_num, -1, -1)

    return probs, masks


def build_additional_mention_to_antecedent(doc_indices, mention_to_antecedent):
    # Note: mention_to_antecedent is a numpy array
    clusters, mention_to_cluster = [], {}
    for mention, antecedent in mention_to_antecedent:
        mention, antecedent = tuple(mention), tuple(antecedent)
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            if mention not in clusters[cluster_idx]:
                clusters[cluster_idx].append(mention)
                mention_to_cluster[mention] = cluster_idx
        elif mention in mention_to_cluster:
            cluster_idx = mention_to_cluster[mention]
            if antecedent not in clusters[cluster_idx]:
                clusters[cluster_idx].append(antecedent)
                mention_to_cluster[antecedent] = cluster_idx
        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])

    clusters = [list(cluster) for cluster in clusters]

    list_new_pairs = []

    for _cluster in clusters:
        if len(_cluster) > 2:
            list_new_pairs.extend([[list(item1), list(item2)] for item1 in _cluster for item2 in _cluster if item1 > item2])

    results = np.concatenate((mention_to_antecedent, list_new_pairs), axis=0)
    results = np.unique(results, axis=0)        
    doc_indices_results = np.pad(doc_indices, (doc_indices[0], results.shape[0] - doc_indices.shape[0]), 'constant')

    return clusters

def enhance_mention_to_antecedent(doc_indices, mention_to_antecedent, mention_indices, antecedent_indices):
    # Note: mention_to_antecedent is a numpy array

    
    unique_doc_indices = np.unique(doc_indices)

    batched_mention_to_antecedent = {idx: mention_to_antecedent[np.nonzero(doc_indices == idx)] for idx in unique_doc_indices}
    batched_mention_indices = {idx: mention_indices[np.nonzero(doc_indices == idx)] for idx in unique_doc_indices}
    batched_antecedent_indices = {idx: antecedent_indices[np.nonzero(doc_indices == idx)] for idx in unique_doc_indices}

    # batched_pair_existence = {
    #     key: set(map(tuple, pairs.reshape(-1, 2))) 
    #     for key, pairs in batched_mention_to_antecedent.items()
    # }
    batched_pair_existence = {
        key: {(tuple(mention), tuple(antecedent)): True for mention, antecedent in batch}
        for key, batch in batched_mention_to_antecedent.items()
    }

    batched_antecedent_to_index = {}
    batched_mention_to_index = {}
    keys = list(batched_mention_indices.keys())

    rs_doc_indices, rs_mention_to_antecedent, rs_mention_indices, rs_antecedent_indices = [], [], [], []

    for key in keys:
        mentions = np.vstack(batched_mention_to_antecedent[key][:, 0])
        antecedents = np.vstack(batched_mention_to_antecedent[key][:, 1])
        
        mention_indices = batched_mention_indices[key]
        antecedent_indices = batched_antecedent_indices[key]
        
        mention_dict = {tuple(mentions[i]): mention_indices[i] for i in range(mentions.shape[0])}
        antecedent_dict = {tuple(antecedents[i]): antecedent_indices[i] for i in range(antecedents.shape[0])}
        
        batched_mention_to_index[key] = mention_dict
        batched_antecedent_to_index[key] = antecedent_dict

    for i in unique_doc_indices:
        # _mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
        # _mention_indices = mention_indices[np.nonzero(doc_indices == i)]
        # _antecedent_indices = antecedent_indices[np.nonzero(doc_indices == i)]

        clusters, mention_to_cluster = [], {}
        for mention, antecedent in batched_mention_to_antecedent[i]:
            mention, antecedent = tuple(mention), tuple(antecedent)
            if antecedent in mention_to_cluster:
                cluster_idx = mention_to_cluster[antecedent]
                if mention not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(mention)
                    mention_to_cluster[mention] = cluster_idx
            elif mention in mention_to_cluster:
                cluster_idx = mention_to_cluster[mention]
                if antecedent not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(antecedent)
                    mention_to_cluster[antecedent] = cluster_idx
            else:
                cluster_idx = len(clusters)
                mention_to_cluster[mention] = cluster_idx
                mention_to_cluster[antecedent] = cluster_idx
                clusters.append([antecedent, mention])

        clusters = [list(cluster) for cluster in clusters]

        list_new_pairs = []
        list_new_mention_indices = []
        list_new_antecedent_indices = []
        for _cluster in clusters:
            if len(_cluster) > 2:
                list_new_pairs.extend([[list(item1), list(item2)] for item1 in _cluster for item2 in _cluster if item1 > item2 and (item1, item2) not in batched_pair_existence[i]])
                
        list_new_mention_indices.extend([batched_mention_to_index[i].get(tuple(item[0])) for item in list_new_pairs]) 
        list_new_antecedent_indices.extend([batched_antecedent_to_index[i].get(tuple(item[1])) for item in list_new_pairs])

        results = np.concatenate((batched_mention_to_antecedent[i], list_new_pairs), axis=0)
        rs_mention_to_antecedent.append(results)
        rs_doc_indices.append(np.full(results.shape[0], i))
        rs_mention_indices.append(np.concatenate((batched_mention_indices[i], list_new_mention_indices), axis=0))
        rs_antecedent_indices.append(np.concatenate((batched_antecedent_indices[i], list_new_antecedent_indices), axis=0))
        
   
    return np.concatenate(rs_doc_indices), np.concatenate(rs_mention_to_antecedent), np.concatenate(rs_mention_indices), np.concatenate(rs_antecedent_indices)

def normalized_probability(batched_M, alpha=0.5, T=1.0):
    batch_size, N, _ = batched_M.size()
    
    # Smoothing with uniform prior
    uniform_prior = torch.full((batch_size, N, N), 1.0/N, device=batched_M.device)
    M_smoothed = (1 - alpha) * batched_M + alpha * uniform_prior
    
    # Softmax with temperature
    P = F.softmax(M_smoothed / T, dim=-1)

    return P