import torch
from transformers import BartTokenizerFast

from transformers_framework.architectures.bart.modeling_bart import (
    BartForMultiTokenConditionalGeneration,
    BartMultiTokenConfig,
)


tok = BartTokenizerFast.from_pretrained('facebook/bart-base')

config = BartMultiTokenConfig.from_pretrained('facebook/bart-base', max_multi_token_predictions=3)
model = BartForMultiTokenConditionalGeneration.from_pretrained('facebook/bart-base', config=config)

w = torch.diag(torch.ones(size=(768,)))
w = torch.cat([w, w, w], axis=0)
model.projection.weight.data = w

inputs = tok("Translate from German to Italian: wie geht es dir?", return_tensors='pt')

res = model.generate(inputs.input_ids, min_length=70, max_length=100, do_sample=False, num_beams=1, num_beam_groups=1)
print(tok.decode(res[0]))

res = model.generate(inputs.input_ids, min_length=70, max_length=100, do_sample=True, num_beams=1, num_beam_groups=1)
print(tok.decode(res[0]))

res = model.generate(inputs.input_ids, min_length=70, max_length=100, do_sample=False, num_beams=4, num_beam_groups=1)
print(tok.decode(res[0]))

res = model.generate(inputs.input_ids, min_length=70, max_length=100, do_sample=True, num_beams=4, num_beam_groups=1)
print(tok.decode(res[0]))

res = model.generate(inputs.input_ids, min_length=70, max_length=100, do_sample=False, num_beams=4, num_beam_groups=2)
print(tok.decode(res[0]))
