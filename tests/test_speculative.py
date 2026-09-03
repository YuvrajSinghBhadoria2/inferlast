"""Tests for speculative (assisted) decode measurement (speculative.py)."""

import torch

from conftest import make_mini_lm
from speculative import measure_speculative


class KVCompat:
    """A causal LM whose forward accepts KV/assisted kwargs, recomputing
    internally -- exercises the measurement plumbing (which is the point)."""

    def __init__(self, vocab=32):
        self.inner = make_mini_lm(vocab=vocab)
        self.embed_tokens = self.inner.embed_tokens
        self.blocks = self.inner.blocks
        self.lm_head = self.inner.lm_head
        self.config = type("C", (), {"eos_token_id": 1, "pad_token_id": 1})()

    def eval(self):
        return self

    def forward(self, ids, attention_mask=None, past_key_values=None,
                use_cache=False, position_ids=None, **kwargs):
        return self.inner(ids)

    def generate(self, input_ids=None, max_new_tokens=8, generation_config=None,
                 return_dict_in_generate=False, assistant_model=None,
                 tokenizer=None, assistant_tokenizer=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            nxt = self.forward(cur).logits[:, -1, :].argmax(-1)
            cur = torch.cat([cur, nxt.unsqueeze(0)], dim=1)
        return type("Out", (), {"sequences": cur})()


def _model(vocab=32):
    return KVCompat(vocab=vocab)


def test_measure_speculative_returns_honest_schema():
    target = _model()
    draft = _model(vocab=32)
    ids = torch.tensor([[0, 1, 2, 3]])
    res = measure_speculative(target, ids, prompt_len=4, assistant_model=draft,
                              n_new=6, repeats=2, warmup=1)
    assert res["avail"] is True
    assert res["plain_kv_tok_s"] > 0
    assert res["assisted_tok_s"] > 0
    assert res["speedup_x"] > 0
    assert res["verdict"] in ("FASTER", "SLOWER", "FLAT")


def test_measure_speculative_has_schema_keys():
    target = _model()
    draft = _model()
    ids = torch.tensor([[0, 1, 2, 3]])
    res = measure_speculative(target, ids, prompt_len=4, assistant_model=draft,
                              n_new=4, repeats=1, warmup=0)
    for k in ("avail", "plain_kv_tok_s", "assisted_tok_s", "speedup_x",
              "verdict", "note"):
        assert k in res, f"missing {k}"
