from __future__ import annotations

import copy
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Literal

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM  # Наследуемся от оригинального класса
from lm_eval.models.utils import pad_and_concat
from lm_eval.api.instance import Instance
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    pad_and_concat,
    postprocess_generated_text,
    stop_sequences_criteria,
)


eval_logger = logging.getLogger(__name__)


# Регистрируем новую, упрощенную модель под другим именем
@register_model("hf-auto-simplified")
class SimpleHFLM(HFLM):
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        print("custom _loglikelihood_tokens")
        # len(requests)=9500
        # requests[0]=[[questions str, answer str], [question tokens int], [answer tokens int]]
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key to group and lookup one-token continuations."""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=(
                "contexts" if self.backend == "causal" and self.logits_cache else None
            ),
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs if override_bs is not None else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice
                # https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777541920

                # when too long to fit in context, truncate from the left
                if self.backend == "causal":
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        eval_logger.warning(
                            f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                            f"exceeds model's maximum length ({self.max_length}). "
                            f"Truncating {total_length - self.max_length + 1} tokens from the left."
                        )
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                # elif self.backend == "seq2seq":
                #     inp = torch.tensor(
                #         (context_enc)[-self.max_length :],
                #         dtype=torch.long,
                #         device=self.device,
                #     )
                #     (inplen,) = inp.shape

                #     # build encoder attn masks
                #     encoder_attns.append(torch.ones_like(inp))

                #     cont = torch.tensor(
                #         (continuation_enc)[-self.max_length :],
                #         # TODO: left-shift these?
                #         # TODO: our code assumes we never end up truncating conts for either model type
                #         dtype=torch.long,
                #         device=self.device,
                #     )
                #     (contlen,) = cont.shape

                #     conts.append(cont)

                #     padding_len_cont = (
                #         max(padding_len_cont, contlen)
                #         if padding_len_cont is not None
                #         else contlen
                #     )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.backend == "causal":
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            # elif self.backend == "seq2seq":
            #     # TODO: left-pad encoder inps and mask?
            #     batched_inps = pad_and_concat(
            #         padding_len_inp, inps
            #     )  # [batch, padding_len_inp]
            #     batched_conts = pad_and_concat(
            #         padding_len_cont, conts
            #     )  # [batch, padding_len_cont]
            #     batched_encoder_mask = pad_and_concat(
            #         padding_len_inp, encoder_attns
            #     )  # [batch, padding_len_inp]
            #     call_kwargs = {
            #         "attn_mask": batched_encoder_mask,
            #         "labels": batched_conts,
            #     }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.backend == "causal"
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(  # noqa
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (
                        greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks
                    ).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        print("custom generate_until")
        res = []

        def _collate(req: tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size if adaptive_batch_size is not None else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise TypeError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs:
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert (
                    max_ctx_len > 0
                ), f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
            # elif self.backend == "seq2seq":
            #     # max len for inputs = encoder's whole max_length
            #     max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1] :]

                # Handle integer think_end_token: find last occurrence and strip tokens after it
                if isinstance(self.think_end_token, int):
                    think_token_indices = [
                        i
                        for i, token in enumerate(cont_toks)
                        if token == self.think_end_token
                    ]
                    if think_token_indices:
                        cont_toks = cont_toks[think_token_indices[-1] + 1 :]

                s = self.tok_decode(cont_toks)

                # Strip leading whitespace if we removed thinking tokens
                if isinstance(self.think_end_token, int):
                    s = s.lstrip()

                # Apply post-processing: remove stop sequences and string-based thinking tokens
                s = postprocess_generated_text(
                    generation=s,
                    stop=until,
                    think_end_token=(
                        self.think_end_token
                        if isinstance(self.think_end_token, str)
                        else None
                    ),
                )
                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        contlen: int | None = None,
        inplen: int | None = None,
    ) -> torch.Tensor:
        if self.backend == "causal":
            assert (
                contlen and inplen
            ), "Must pass input len and cont. len to select scored logits for causal LM"
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        # elif self.backend == "seq2seq":
        #     assert (
        #         contlen and not inplen
        #     ), "Selecting scored logits for Seq2SeqLM requires only cont. len"
        #     # only discard right-padding.
        #     # the logits input to this fn only contain decoder-side tokens.
        #     logits = logits[:contlen]

        return logits

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )
