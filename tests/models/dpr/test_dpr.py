# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/dpr-reader-single-nq-base

from transformers import DPRReader, DPRReaderTokenizer
import pytest
from tests.utils import ModelTester, repeat_inputs
import torch


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", torch_dtype=torch.bfloat16)
        return model

    def _load_inputs(self, batch_size):
        encoded_inputs = self.tokenizer(
            questions=["What is love ?"],
            titles=["Haddaway"],
            texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            return_tensors="pt",
        )
        encoded_inputs = repeat_inputs(encoded_inputs, batch_size)
        return encoded_inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.converted_end_to_end
def test_dpr(record_property, mode):
    model_name = "DPR"
    record_property("model_name", model_name)
    record_property("mode", mode)

    tester = ThisTester(model_name, mode)
    results = tester.test_model()

    if mode == "eval":
        start_logits = results.start_logits
        end_logits = results.end_logits
        relevance_logits = results.relevance_logits
        print(results)

    record_property("torch_ttnn", (tester, results))
