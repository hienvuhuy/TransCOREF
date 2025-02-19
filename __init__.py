# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .models.transcoref.transcoref_config import (
    TransCOREFConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .models.transcoref.transcoref_decoder import TransCOREFDecoder, TransCOREFDecoderBase, Linear
from .models.transcoref.transcoref_encoder import TransCOREFEncoder, TransCOREFEncoderBase


from .models.transcoref.transcoref_models import *
from .models.transcoref.transcoref_base import TransCOREFModelBase, Embedding
from .criterions.label_smoothed_cross_entropy_with_coref import *


# from .criterions.label_smoothed_cross_entropy_with_coref_guided import *
# from .criterions.dependency_modeling_criterion import *
# from .criterions.dependency_translation_modeling_criterion import *
# # from .criterions.dependency_translation_modeling_criterion_doc import *
# from .criterions.dependency_translation_with_coref_modeling_criterion import *
# from .criterions.label_smoothed_cross_entropy_debug import *
# from .criterions.label_smoothed_cross_entropy_join_training import *
# # from .transformer_coref_guided_lm_layer import TransformerCorefGuidedLanguageModelConfig, TransformerCorefGuidedLanguageModel
# from .models.coref_guided.transformer_coref_guided_lm import TransformerCorefGuidedLanguageModelConfig, TransformerCorefGuidedLanguageModel
# from .models.coref_guided.transformer_coref_guided_lm import *

__all__ = [
    "TransformerCorefProbingModelBase",
    "TransCOREFConfig",
    "TransformerCorefProbingDecoder",
    "TransformerCorefProbingDecoderBase",
    "TransCOREFEncoder",
    "TransCOREFEncoderBase",
    "TransformerCorefProbingModel",
    "TransformerCorefProbingLanguageModel"
    "Embedding",
    "Linear",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
    # "TranslationCorefTask"
]
