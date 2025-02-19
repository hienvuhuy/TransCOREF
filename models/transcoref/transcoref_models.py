# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import (
    register_model,
    register_model_architecture,
)

# from fairseq.models.transformer_coref_probing.transformer_coref_probing_config import (
#     TransformerCorefProbingConfig,
#     DEFAULT_MAX_SOURCE_POSITIONS,
#     DEFAULT_MAX_TARGET_POSITIONS,
#     DEFAULT_MIN_PARAMS_TO_WRAP,
# )

from TransCOREF.models.transcoref.transcoref_config import (
    TransCOREFConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)

# from fairseq.models.transformer_coref_probing.transformer_coref_probing_base import (
#     TransformerCorefProbingModelBase,
# )

from TransCOREF.models.transcoref.transcoref_base import (
    TransCOREFModelBase,
)

from omegaconf import DictConfig, open_dict, OmegaConf


def __getattr(__o: object, name: str, __default: None):
    # Since the default getattr function does not work well with the DictConfig object, this 
    #   alternative function is to hack this damn thing
    if getattr(__o, name, __default) is None:
        return __default
    return getattr(__o, name, __default)

# architectures
def _base_architecture(args):
    args.encoder_embed_path = __getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = __getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = __getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = __getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = __getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = __getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = __getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = __getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = __getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = __getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = __getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = __getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = __getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = __getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = __getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = __getattr(args, "activation_dropout", 0.1)
    args.activation_fn = __getattr(args, "activation_fn", "relu")
    args.dropout = __getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = __getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = __getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = __getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = __getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = __getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = __getattr(args, "adaptive_input", False)
    args.no_cross_attention = __getattr(args, "no_cross_attention", False)
    args.cross_self_attention = __getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = __getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = __getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = __getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = __getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = __getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = __getattr(args, "checkpoint_activations", False)
    args.offload_activations = __getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = __getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = __getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = __getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = __getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = __getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = __getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = __getattr(args, "quant_noise_scalar", 0)

# A simple hack to pass the immature of the fairseq's hydra-training 
def base_architecture(args):
    if isinstance(args, DictConfig):
        with open_dict(args):
            _base_architecture(args)
        # pass
    else:
        _base_architecture(args)

@register_model("transformer_coref_probing", dataclass=TransCOREFConfig)
class TransCOREFModel(TransCOREFModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        def spm(path):
            return {
                'path': path,
                'bpe': 'sentencepiece',
                'tokenizer': 'space',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            # 'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            # 'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            # 'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            # 'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            # 'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            # 'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            # 'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            # 'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            # 'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            # 'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
            # 'transformer.wmt20.en-ta': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz'),
            # 'transformer.wmt20.en-iu.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz'),
            # 'transformer.wmt20.en-iu.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz'),
            # 'transformer.wmt20.ta-en': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz'),
            # 'transformer.wmt20.iu-en.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz'),
            # 'transformer.wmt20.iu-en.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz'),
            # 'transformer.flores101.mm100.615M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz'),
            # 'transformer.flores101.mm100.175M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder, coref_representation=None, coref_module=None):
        cfg = TransCOREFConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.coref_representation = coref_representation
        self.coref_module = coref_module
        self.args = args

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransCOREFConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            TransCOREFConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            TransCOREFConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            TransCOREFConfig.from_namespace(args), tgt_dict, embed_tokens
        )



