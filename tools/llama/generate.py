import os
import time
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.text.parser import clean_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from fish_speech.models.text2semantic.llama import Transformer
from fish_speech.text import g2p
from fish_speech.text.symbols import pad as pad_symbol
from fish_speech.text.symbols import pu_symbols


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    assert input_pos.shape[-1] == 1

    logits = model.forward_generate(x, input_pos)

    return sample(
        logits,
        previous_tokens=previous_tokens,
        **sampling_kwargs,
    )[0]


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    logits = model.forward_generate(x, input_pos)
    return sample(
        logits,
        previous_tokens=None,
        **sampling_kwargs,
    )[0]


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    eos_token_id: int = 2,
    **sampling_kwargs,
):
    previous_tokens = torch.zeros(
        model.config.max_seq_len,
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:win_size]
        else:
            window = previous_tokens[i - win_size : i]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(1, -1)
        previous_tokens[i : i + 1] = next_token.view(-1)

        # TODO: use tokenizer's eos
        if cur_token[0, -1] == eos_token_id:
            break

    return previous_tokens[: i + 1]


@torch.no_grad()
def generate(
    *,
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int = 2,
    precision: torch.dtype = torch.bfloat16,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_len=T_new, dtype=precision)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        max_new_tokens - 1,
        eos_token_id=eos_token_id,
        **sampling_kwargs,
    )

    seq = seq[: T + 1 + x.size(0)]
    seq[T + 1 :] = x

    return seq


def encode_tokens(
    tokenizer,
    string,
    bos=True,
    device="cuda",
    prompt_tokens=None,
    use_g2p=False,
    speaker=None,
    order="zh,jp,en",
):
    if use_g2p:
        order = order.split(",")
        prompt = g2p(string, order=order)
        prompt = [
            (f"<p:{i}>" if i not in pu_symbols and i != pad_symbol else i)
            for _, i in prompt
        ]
        string = "".join(prompt)
    else:
        string = clean_text(string)

    if speaker is not None:
        string = f"[SPK: {speaker}] {string}"

    string = f"[INST] {string} [/INST]"

    new_tokens = tokenizer.encode(
        string,
        add_special_tokens=bos,
        max_length=10**6,
        truncation=False,
    )
    prompt = torch.tensor(new_tokens, dtype=torch.int, device=device)

    if prompt_tokens is None:
        return prompt

    # Get prompt tokens
    if prompt_tokens.ndim == 2:
        assert (
            prompt_tokens.shape[0] == 1
        ), f"3 dim prompt tokens should have shape (1, seq_len)"
        prompt_tokens = prompt_tokens[0]

    assert prompt_tokens.ndim == 1
    prompt_tokens = [f"<s:{i}>" for i in prompt_tokens]
    prompt_tokens = tokenizer.convert_tokens_to_ids(prompt_tokens)
    prompt = torch.cat((prompt, prompt_tokens), dim=0)

    return prompt


def load_model(config_name, checkpoint_path, device, precision):
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    # with torch.device("meta"):
    base_model = instantiate(cfg.model)
    ar_model: Transformer = base_model.ar_model
    nar_model: Transformer = base_model.nar_model

    if "int8" in str(checkpoint_path):
        logger.info("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(ar_model)
        ar_model = simple_quantizer.convert_for_runtime()

        simple_quantizer = WeightOnlyInt8QuantHandler(nar_model)
        nar_model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    ar_checkpoint = {
        k.replace("ar_model.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("ar_model.")
    }

    nar_checkpoint = {
        k.replace("nar_model.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("nar_model.")
    }

    ar_model.load_state_dict(ar_checkpoint, assign=True, strict=True)
    nar_model.load_state_dict(nar_checkpoint, assign=True, strict=True)

    ar_model = ar_model.to(device=device, dtype=precision)
    nar_model = nar_model.to(device=device, dtype=precision)
    logger.info("Restored model from checkpoint")

    return ar_model.eval(), nar_model.eval(), cfg


def split_text(text, min_length):
    text = clean_text(text)
    segments = []
    curr = ""
    for char in text:
        curr += char
        if char not in [".", ",", "!", "?"]:
            continue

        if len(curr) >= min_length:
            segments.append(curr)
            curr = ""

    if curr:
        segments.append(curr)

    return segments


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None)
@click.option(
    "--prompt-tokens", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-k", type=int, default=None)
@click.option("--top-p", type=float, default=0.5)
@click.option("--repetition-penalty", type=float, default=1.5)
@click.option("--temperature", type=float, default=0.7)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="results/text2semantic_400m_finetune/step_000002000.pth",
)
@click.option("--config-name", type=str, default="text2semantic_finetune")
@click.option("--tokenizer", type=str, default="fishaudio/speech-lm-v1")
@click.option("--compile/--no-compile", default=False)
@click.option("--use-g2p/--no-g2p", default=True)
@click.option("--seed", type=int, default=42)
@click.option("--speaker", type=str, default=None)
@click.option("--order", type=str, default="zh,jp,en")
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=False)
def main(
    text: str,
    prompt_text: Optional[str],
    prompt_tokens: Optional[Path],
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    config_name: str,
    tokenizer: str,
    compile: bool,
    use_g2p: bool,
    seed: int,
    speaker: Optional[str],
    order: str,
    half: bool,
    iterative_prompt: bool,
) -> None:
    device = "cuda"

    precision = torch.half if half else torch.bfloat16

    logger.info("Loading model ...")
    t0 = time.time()
    ar_model, nar_model, cfg = load_model(
        config_name, checkpoint_path, device, precision
    )
    ar_model_size = sum(p.numel() for p in ar_model.parameters() if p.requires_grad)

    torch.cuda.synchronize()
    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision="yi-34b")
    semantic_tokens = [f"<s:{i}>" for i in range(1024)]
    codebook_tokens = tokenizer.convert_tokens_to_ids([f"<c:{i}>" for i in range(16)])

    semantic_tokens_lookup = {
        a: b for b, a in enumerate(tokenizer.convert_tokens_to_ids(semantic_tokens))
    }

    prompt_tokens = (
        torch.from_numpy(np.load(prompt_tokens)).to(device)
        if prompt_tokens is not None
        else None
    )

    use_prompt = prompt_text is not None and prompt_tokens is not None
    encoded = []
    texts = split_text(text, 20) if iterative_prompt else [text]
    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                bos=idx == 0 and not use_prompt,
                device=device,
                use_g2p=use_g2p,
                speaker=None,
                order=order,
            )
        )
        print(f"Encoded text: {text}")

    if use_prompt and iterative_prompt:
        encoded_prompt = encode_tokens(
            tokenizer,
            prompt_text,
            prompt_tokens=prompt_tokens,
            bos=True,
            device=device,
            use_g2p=use_g2p,
            speaker=speaker,
            order=order,
        )

        encoded[0] = torch.cat((encoded_prompt, encoded[0]), dim=1)

    # prompt_length = encoded.size(1)
    # logger.info(f"Encoded prompt shape: {encoded.shape}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if compile:
        global decode_one_token
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

    for idx in range(num_samples):
        torch.cuda.synchronize()
        global_encoded = []
        all_codes = []
        seg_idx = 0

        while seg_idx < len(encoded):
            seg = encoded[seg_idx]
            global_encoded.append(seg)
            cat_encoded = torch.cat(global_encoded, dim=0)
            prompt_length = cat_encoded.size(0)

            t0 = time.perf_counter()
            y = generate(
                model=ar_model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            torch.cuda.synchronize()
            t = time.perf_counter() - t0

            tokens_generated = y.size(0) - prompt_length
            tokens_sec = tokens_generated / t
            logger.info(
                f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {ar_model_size * tokens_sec / 1e9:.02f} GB/s"
            )
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

            # Put the generated tokens
            codes = y[prompt_length:-1].clone()
            codes = [semantic_tokens_lookup.get(i.item(), -1) for i in codes]
            codes = torch.tensor(codes, dtype=torch.int)
            assert (codes != -1).all(), f"Unknown code found: {codes}"

            global_encoded.append(y[prompt_length:-1].clone())
            all_codes.append(codes)
            seg_idx += 1

        codes = torch.cat(all_codes, dim=0)
        assert (codes >= 0).all(), f"Negative code found: {codes}"
        # codes = [codes]

        # Now it's time to handle NAR part
        global_encoded = torch.cat(global_encoded, dim=0)
        token_mask = [
            idx
            for idx, i in enumerate(global_encoded)
            if int(i) in semantic_tokens_lookup
        ]
        token_mask = torch.tensor(token_mask, dtype=torch.int)

        codes = [
            torch.tensor(
                [
                    semantic_tokens_lookup.get(i.item(), None)
                    for i in global_encoded[token_mask]
                ],
                dtype=torch.int,
            ),
        ]

        for i in range(1, 8):
            temp = global_encoded.clone()
            temp[0] = codebook_tokens[i]

            logits = nar_model.forward(
                temp.unsqueeze(0),
            )

            # Get argmax
            tokens = logits.argmax(dim=-1)[0][token_mask]
            tokens = [semantic_tokens_lookup.get(i.item(), -1) for i in tokens]
            tokens = torch.tensor(tokens, dtype=torch.int)
            assert (tokens != -1).all(), f"Unknown code found: {tokens}"

            codes.append(tokens)
            logger.info(f"Generated codebook {i}")

        codes = torch.stack(codes, dim=0)
        print(codes.shape, codes)

        np.save(f"codes_{idx}.npy", codes.cpu().numpy())
        logger.info(f"Saved codes to codes_{idx}.npy")


if __name__ == "__main__":
    main()
