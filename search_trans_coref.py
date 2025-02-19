# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional

import torch
import torch.nn as nn

from torch import Tensor
from fairseq import search 
class CorefBeamSearch(search.BeamSearch):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        # self.constraint_states = None

    @torch.jit.export
    def future_step(
        self,
        step: int,
        lprobs,
        scores: Optional[Tensor],
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
        number_future_tokens=None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        if number_future_tokens is None:
            top_prediction = torch.topk(
                lprobs.view(bsz, -1),
                k=min(
                    # Take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.
                    beam_size * 2,
                    lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
                ),
            )
        else:
            # top_prediction = torch.topk(
            #     lprobs.view(bsz, -1),
            #     k=number_future_tokens,
            # )

            # We select #number_future_tokens of each hypothesis to help generating temporaturely sequences
            top_prediction = torch.topk(
                lprobs,
                k=number_future_tokens
            )
            return top_prediction[0], top_prediction[1]

        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = torch.div(indices_buf, vocab_size, rounding_mode="trunc")
        indices_buf = indices_buf.fmod(vocab_size)

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf
