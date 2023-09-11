import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

import llama_cpp
from guidance.llms import LLM, LLMSession
from llama_cpp import Completion, Llama, StoppingCriteriaList
from loguru import logger

logger.disable("llama_cpp_guidance")


class LlamaCppTokenizer:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.cache = {}

    def encode(self, string: str, **kwargs) -> List[int]:
        logger.trace("Encoding string: {string}", string=string)
        if string in self.cache:
            logger.debug(
                "Cache hit `{string}` => `{token}`",
                string=string,
                token=self.cache[string],
            )
            return self.cache[string]

        tokens = self.llm.tokenize(string.encode("utf-8"), **kwargs)

        self.cache[string] = tokens

        return tokens

    def decode(self, tokens, **kwargs) -> str:
        logger.trace("Decoding tokens: {tokens}", tokens=tokens)
        return self.llm.detokenize(tokens, **kwargs).decode("utf-8")


class LlamaCpp(LLM):
    """A LlamaCpp LLM class for Guidance."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 1024,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        role_start_tag="<|im_start|>",
        role_end_tag="<|im_end|>",
        chat_mode=False,
        seed: int = 0,
        role_to_name: Dict[str, str] = {},
        **llama_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.llm_name = "llama-cpp"
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.role_start_tag = role_start_tag
        self.role_end_tag = role_end_tag
        self.chat_mode = chat_mode
        self.role_to_name = role_to_name

        logger.debug(f"Instantiating LlamaCpp ({model_path})")

        self.llm = Llama(
            model_path=str(model_path),
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,
            verbose=False,
            seed=seed,
            **llama_kwargs,
        )
        logger.debug("Llama instantiated")
        self._tokenizer = LlamaCppTokenizer(self.llm)

    def session(self, asynchronous=False):
        """Creates a session for the LLM."""

        if asynchronous:
            return LlamaCppSession(self)
        else:
            raise NotImplementedError

    def _call_llm(self, *args, **kwargs) -> Completion:
        """Internal call of the Llama LLM model."""
        logger.debug("Invoking LlamaCpp ({args}) ({kwargs})", args=args, kwargs=kwargs)

        llm_out = self.llm(*args, **kwargs)

        logger.debug(
            "LlamaCpp response: {output} ({type})", output=llm_out, type=type(llm_out)
        )

        if not isinstance(llm_out, Iterator):
            return llm_out

        logger.debug("Iterator detected! {content}", content=llm_out)
        completion_chunks = list(llm_out)

        completion = completion_chunks[0]

        for chunk in completion_chunks[1:-1]:
            for index, choice in enumerate(chunk.get("choices", [])):
                completion["choices"][index]["text"] += choice["text"]
                completion["choices"][index]["finish_reason"] = choice["finish_reason"]

        logger.debug("Merged completion chunks. {completion}", completion=completion)

        return completion

    def __call__(self, *args, **kwargs) -> Completion:
        output: Completion = self._call_llm(*args, **kwargs)

        for choice in output.get("choices", []):
            logprobs = choice.get("logprobs")

            if not logprobs:
                continue

            new_top_logprobs = []
            for index, top_logprobs in enumerate(logprobs["top_logprobs"]):
                if top_logprobs is None:
                    top_logprobs = {choice["logprobs"]["tokens"][index]: -0.01}

                new_top_logprobs.append(top_logprobs)
            logprobs["top_logprobs"] = new_top_logprobs

        return output

    def token_to_id(self, text):
        ids = self.encode(text, add_bos=False)

        return ids[-1]

    def role_start(self, role_name, **kwargs):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return (
            self.role_start_tag
            + self.role_to_name.get(role_name, role_name)
            + "".join([f' {k}="{v}"' for k, v in kwargs.items()])
            + "\n"
        )

    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return self.role_end_tag

    def end_of_text(self):
        return "[end of text]"


class LlamaCppSession(LLMSession):
    """A session handler for LlamaCpp"""

    def make_logit_bias_processor(
        self,
        logit_bias: Dict[str, float],
        logit_bias_type: Optional[Literal["input_ids", "tokens"]],
    ):
        if logit_bias_type is None:
            logit_bias_type = "input_ids"

        to_bias: Dict[int, float] = {}
        if logit_bias_type == "input_ids":
            for input_id, score in logit_bias.items():
                input_id = int(input_id)
                to_bias[input_id] = score

        elif logit_bias_type == "tokens":
            for token, score in logit_bias.items():
                token = token.encode("utf-8")
                for input_id in self.llm.tokenize(token, add_bos=False):
                    to_bias[input_id] = score

        def logit_bias_processor(
            input_ids: List[int],
            scores: List[float],
        ) -> List[float]:
            new_scores = [None] * len(scores)
            for input_id, score in enumerate(scores):
                new_scores[input_id] = score + to_bias.get(input_id, 0.0)

            return new_scores

        return logit_bias_processor

    async def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs,
    ):
        logits_processor = None
        if "logit_bias" in kwargs and kwargs["logit_bias"] is not None:
            # Logits are the options we want. Cache the tokens so we can enforce their
            # usage during token_to_id.
            for id in kwargs["logit_bias"].keys():
                token = self.llm.decode([id])
                self.llm._tokenizer.cache[token] = [id]

            logits_processor = llama_cpp.LogitsProcessorList(
                [
                    self.make_logit_bias_processor(kwargs["logit_bias"], "input_ids"),
                ]
            )

        return self.llm(
            prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
        )
