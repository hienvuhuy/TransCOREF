# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os, ast
from typing import Optional
from argparse import Namespace
from webbrowser import get

from omegaconf import II
import torch
from collections import Counter

import numpy as np
from fairseq import metrics, search, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    # LanguagePairDatasetWithCoref,
    PrependTokenDataset,
    StripTokenDataset,
    RawLabelDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
# from fairseq.
from scipy.optimize import linear_sum_assignment
import einops
from TransCOREF.sequence_generator_trans_coref import SequenceGeneratorTransCoref

from TransCOREF.utils import trans_coref_utils
from TransCOREF.data.language_pair_dataset_coref import LanguagePairDatasetWithCoref


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


# pre-defined classes for Coref module
def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, 

def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p

def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = sum(scores[row_ind, col_ind])
    return similarity, len(clusters), similarity, len(gold_clusters)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt, 
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    load_source_coref=False,
    load_target_coref=False,
    coref_cluster_path='',
    coref_cluster_path_test_split=None, # hien-v: check here if errors
    coref_configs=None,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )
    
    src_coref = None
    if load_source_coref:
        if coref_cluster_path:
            if split == "test":
                if coref_cluster_path_test_split:
                    src_coref_path = coref_cluster_path_test_split 
                    #os.path.join(coref_cluster_path, "{}.{}-{}.{}.coref".format(split, src, tgt, src))
                else:
                    src_coref_path = os.path.join(coref_cluster_path, "{}.{}-{}.{}.coref".format(split, src, tgt, src))
            else:
                src_coref_path =   os.path.join(coref_cluster_path, "{}.{}-{}.{}.coref".format(split, src, tgt, src))
        if os.path.exists(src_coref_path):
            src_coref_info = []
            corefs = [i.strip() for i in open(src_coref_path, 'r').readlines()]
            if coref_configs:
                max_clusters, max_items_in_cluster = coref_configs
            else:
                max_clusters, max_items_in_cluster = 8, 16
            src_coref_info = trans_coref_utils.packed_list_data(corefs, max_clusters , max_items_in_cluster,return_tensor=False)

            src_coref = RawLabelDataset(src_coref_info)
    
    if load_target_coref:
        # For future integration into target side
        pass
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDatasetWithCoref(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        src_coref=src_coref,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class TranslationWithCorefConfig(TranslationConfig):
    load_source_coref: bool = field(
        default=False, metadata = {"help": "load coreference cluster information in source side"}
    )
    load_target_coref: bool = field(
        default=False, metadata = {"help": "load coreference cluster information in targert side"}
    )
    coref_cluster_path: str = field(
        default='', metadata={"help":"path to coref cluster files"}
    )
    coref_cluster_path_test_split: str = field(
        default='', metadata={"help":"path to coref cluster file for ONLY test set"}
    )
    max_clusters: int = field(
        default=8, metadata={"help": "maximum number of cluster in one input. Examine traning set to select this parameters"}
    )
    max_items_in_cluster: int = field(
        default=16, metadata={"help": "maximum number of items in one clusters. Examine traning set to select this parameters"}
    )
    max_tokens_in_cluster: int = field(
        default=65, metadata={"help": "maximum number of items in one clusters. Examine traning set to select this parameters"}
    )
    coref_validation_devset: bool = field(
        default=False, metadata = {"help": "validate accuracies of coref module in dev set"}
    )



@register_task("translation_with_coref", dataclass=TranslationWithCorefConfig)
class TranslationWithCorefTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationWithCorefConfig

    def __init__(self, cfg: TranslationWithCorefConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        if cfg.coref_validation_devset:
            self.coref_evaluator = CorefEvaluator()


    @classmethod
    def setup_task(cls, cfg: TranslationWithCorefConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        coref_configs = (self.cfg.max_clusters, self.cfg.max_items_in_cluster)
        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            load_source_coref=self.cfg.load_source_coref,
            load_target_coref=self.cfg.load_target_coref,
            coref_cluster_path=self.cfg.coref_cluster_path,
            coref_cluster_path_test_split=self.cfg.coref_cluster_path_test_split,
            coref_configs=coref_configs,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )
            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model
    def valid_step_for_consistency_test(self, sample, model, criterion, consistency_test=False):
        model.eval()
        with torch.no_grad():
            if self.cfg.coref_validation_devset:
                loss, sample_size, logging_output, net_output = criterion(model, sample, accuracy=True)    
            else:
                loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output



    def valid_step(self, sample, model, criterion):

        model.eval()
        with torch.no_grad():
            if self.cfg.coref_validation_devset:
                loss, sample_size, logging_output, net_output = criterion(model, sample, accuracy=True)    
            else:
                loss, sample_size, logging_output = criterion(model, sample)
        
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len


            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        if self.cfg.coref_validation_devset:
            _gold_clusters = sample['net_input']['src_coref']
            src_tokens = sample['net_input']['src_tokens']
            
            # add padding 
            coref_shape = _gold_clusters.shape
            pad = torch.sum(src_tokens == self.src_dict.pad_index, dim=1)
            pad = einops.repeat(pad, "h -> h c d e", c=coref_shape[1], d=coref_shape[2], e=coref_shape[-1])
            _gold_clusters = torch.where(_gold_clusters > 0, _gold_clusters + pad, _gold_clusters)

           
            batch_gold_clusters = trans_coref_utils.batch_extract_clusters(_gold_clusters)
            batch_mention_to_gold_clusters = trans_coref_utils.batch_extract_mentions_to_predicted_clusters_from_clusters(batch_gold_clusters)


            batch_starts, batch_end_offsets, batch_coref_logits, \
                batch_mention_logits = net_output['coref_out']['coref_probing_output'][1:-1]

            batch_max_antecedents = torch.argmax(batch_coref_logits, dim=2).tolist()
            batch_mention_to_antecedents =[
                {((int(_start), int(_end)), (int(starts[_max_antecedents]), int(end_offsets[_max_antecedents]))) \
                    for _start, _end, _max_antecedents in zip(starts, end_offsets, max_antecedents) if _max_antecedents < len(starts)}
                    for starts, end_offsets, max_antecedents in zip(batch_starts, batch_end_offsets, batch_max_antecedents)
            ]
            batch_predicted_clusters = [trans_coref_utils.extract_clusters_for_decode(mention_to_antecedent)[0] \
                                            for mention_to_antecedent in batch_mention_to_antecedents] 
            
            batch_mention_to_predicted_clusters = [trans_coref_utils.extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)\
                                                for predicted_clusters in batch_predicted_clusters]

            # muc metrics
            _muc_p_num = 0
            _muc_p_den = 0
            _muc_r_num = 0
            _muc_r_den = 0
            
            muc_p = [muc(predicted_clusters, mention_to_gold_clusters) for predicted_clusters, mention_to_gold_clusters \
                        in zip(batch_predicted_clusters,batch_mention_to_gold_clusters)]
            
            muc_r = [muc(gold_clusters, mention_to_predicted_clusters) for gold_clusters, mention_to_predicted_clusters \
                        in zip(batch_gold_clusters,batch_mention_to_predicted_clusters)]
            
            _muc_p_num = sum(i for i, j in muc_p)
            _muc_p_den = sum(j for i, j in muc_p)
            _muc_r_num = sum(i for i, j in muc_r)
            _muc_r_den = sum(j for i, j in muc_r)

            logging_output["_muc_p_num"] = _muc_p_num
            logging_output["_muc_p_den"] = _muc_p_den

            logging_output["_muc_r_num"] = _muc_r_num
            logging_output["_muc_r_den"] = _muc_r_den



            # only for testing purpose
            coref_correct = 10
            coref_incorrect = int(sample['nsentences']) - 10
            logging_output['_coref_correct'] = coref_correct
            logging_output['_coref_incorrect'] = coref_incorrect
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        # reduce metric and return 
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)
        if self.cfg.coref_validation_devset:
            # key starts that with '_' will be deleted 
            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result
            
            muc_p_nums = sum_logs('_muc_p_num')
            muc_p_dens = sum_logs('_muc_p_den')
            muc_r_nums = sum_logs('_muc_r_num')
            muc_r_dens = sum_logs('_muc_r_den')

            if muc_r_dens >0:
                if muc_r_nums == 0:
                    metrics.log_scalar("coref_muc_recall",0)
                else:
                    metrics.log_scalar("coref_muc_recall",muc_r_nums/float(muc_r_dens))

            if muc_p_dens > 0:
                if muc_p_nums == 0:
                    metrics.log_scalar("coref_muc_precision",0)
                else:
                    metrics.log_scalar("coref_muc_precision",muc_p_nums/float(muc_p_dens))

            if muc_p_dens > 0 and muc_r_dens > 0:
                metrics.log_scalar("_muc_p_den", sum_logs("_muc_p_den"))
                metrics.log_scalar("_muc_p_num", sum_logs("_muc_p_num"))
                metrics.log_scalar("_muc_r_den", sum_logs("_muc_r_den"))
                metrics.log_scalar("_muc_r_num", sum_logs("_muc_r_num"))

                def _f1(meters):
                    beta=1
                    p_den=meters["_muc_p_den"].sum
                    p_num=meters["_muc_p_num"].sum
                    r_den=meters["_muc_r_den"].sum
                    r_num=meters["_muc_r_num"].sum
                    p = 0 if p_den == 0 else p_num / float(p_den)
                    r = 0 if r_den == 0 else r_num / float(r_den)
                    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

                metrics.log_derived("coref_muc_f1", _f1)


    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def _predict_with_coref(self, sample, model):
        pass

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            elif getattr(args, "with_probing_generate", False):
                seq_gen_cls = SequenceGeneratorTransCoref
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )