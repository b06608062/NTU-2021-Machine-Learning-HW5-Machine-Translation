import sys
import pdb
import pprint
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


data_dir = "./DATA/rawdata"
dataset_name = "ted2020"
prefix = Path(data_dir).absolute() / dataset_name


src_lang = "en"
tgt_lang = "zh"

data_prefix = f"{prefix}/train_dev.raw"
test_prefix = f"{prefix}/test.raw"


with open(f"{data_prefix}.{src_lang}", "r", encoding="utf-8") as file:
    for _ in range(5):
        print(file.readline().strip())

with open(f"{data_prefix}.{tgt_lang}", "r", encoding="utf-8") as file:
    for _ in range(5):
        print(file.readline().strip())


import re


def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif (
                inside_code >= 65281 and inside_code <= 65374
            ):  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


def clean_s(s, lang):
    if lang == "en":
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace("-", "")  # remove '-'
        s = re.sub('([.,;!?()"])', r" \1 ", s)  # keep punctuation
    elif lang == "zh":
        s = strQ2B(s)  # Q2B
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace(" ", "")
        s = s.replace("—", "")
        s = s.replace("“", '"')
        s = s.replace("”", '"')
        s = s.replace("_", "")
        s = re.sub('([。,;!?()"~「」])', r" \1 ", s)  # keep punctuation
    s = " ".join(s.strip().split())
    return s


def len_s(s, lang):
    if lang == "zh":
        return len(s)
    return len(s.split())


def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f"{prefix}.clean.{l1}").exists() and Path(f"{prefix}.clean.{l2}").exists():
        print(f"{prefix}.clean.{l1} & {l2} exists. skipping clean.")
        return
    with open(f"{prefix}.{l1}", "r", encoding="utf-8") as l1_in_f:
        with open(f"{prefix}.{l2}", "r", encoding="utf-8") as l2_in_f:
            with open(f"{prefix}.clean.{l1}", "w", encoding="utf-8") as l1_out_f:
                with open(f"{prefix}.clean.{l2}", "w", encoding="utf-8") as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0:  # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0:  # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0:  # remove by ratio of length
                            if s1_len / s2_len > ratio or s2_len / s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)


with open(f"{data_prefix}.clean.{src_lang}", "r", encoding="utf-8") as file:
    for _ in range(5):
        print(file.readline().strip())

with open(f"{data_prefix}.clean.{tgt_lang}", "r", encoding="utf-8") as file:
    for _ in range(5):
        print(file.readline().strip())


valid_ratio = 0.01  # 3000~4000 would suffice
train_ratio = 1 - valid_ratio


if (
    (prefix / f"train.clean.{src_lang}").exists()
    and (prefix / f"train.clean.{tgt_lang}").exists()
    and (prefix / f"valid.clean.{src_lang}").exists()
    and (prefix / f"valid.clean.{tgt_lang}").exists()
):
    print(f"train/valid splits exists. skipping split.")
else:
    line_num = sum(
        1 for line in open(f"{data_prefix}.clean.{src_lang}", "r", encoding="utf-8")
    )
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f = open(
            os.path.join(data_dir, dataset_name, f"train.clean.{lang}"),
            "w",
            encoding="utf-8",
        )
        valid_f = open(
            os.path.join(data_dir, dataset_name, f"valid.clean.{lang}"),
            "w",
            encoding="utf-8",
        )
        count = 0
        for line in open(f"{data_prefix}.clean.{lang}", "r", encoding="utf-8"):
            if labels[count] / line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()


import sentencepiece as spm


vocab_size = 8000
if (prefix / f"spm{vocab_size}.model").exists():
    print(f"{prefix}/spm{vocab_size}.model exists. skipping spm_train.")
else:
    spm.SentencePieceTrainer.train(
        input=",".join(
            [
                f"{prefix}/train.clean.{src_lang}",
                f"{prefix}/valid.clean.{src_lang}",
                f"{prefix}/train.clean.{tgt_lang}",
                f"{prefix}/valid.clean.{tgt_lang}",
            ]
        ),
        model_prefix=prefix / f"spm{vocab_size}",
        vocab_size=vocab_size,
        character_coverage=1,
        model_type="unigram",  # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc_cf",
    )


spm_model = spm.SentencePieceProcessor(
    model_file=str(prefix / f"spm{vocab_size}.model")
)
in_tag = {
    "train": "train.clean",
    "valid": "valid.clean",
    "test": "test.raw.clean",
}
for split in ["train", "valid", "test"]:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix / f"{split}.{lang}"
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix / f"{split}.{lang}", "w", encoding="utf-8") as out_f:
                with open(
                    prefix / f"{in_tag[split]}.{lang}", "r", encoding="utf-8"
                ) as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(" ".join(tok), file=out_f)


with open(f"{data_dir}/{dataset_name}/train.{src_lang}", "r", encoding="utf-8") as file:
    for _ in range(5):
        print(file.readline().strip())

with open(f"{data_dir}/{dataset_name}/train.{tgt_lang}", "r", encoding="utf-8") as file:
    for _ in range(5):
        print(file.readline().strip())


import subprocess


binpath = Path("./DATA/data-bin", dataset_name)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    cmd = [
        "python",
        "-m",
        "fairseq_cli.preprocess",
        "--source-lang",
        src_lang,
        "--target-lang",
        tgt_lang,
        "--trainpref",
        str(prefix / "train"),
        "--validpref",
        str(prefix / "valid"),
        "--testpref",
        str(prefix / "test"),
        "--destdir",
        str(binpath),
        "--joined-dictionary",
        "--workers",
        "2",
    ]
    subprocess.run(cmd, check=True)


config = Namespace(
    datadir="./DATA/data-bin/ted2020",
    savedir="./checkpoints/backward",
    source_lang="zh",
    target_lang="en",
    # cpu threads when fetching & processing data.
    num_workers=0,
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,
    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.0,
    lr_warmup=4000,
    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,
    # maximum epochs for training
    max_epoch=30,
    start_epoch=1,
    # beam size for beam search
    beam=5,
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2,
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process="sentencepiece",
    # checkpoints
    keep_last_epochs=5,
    resume=None,  # if resume from checkpoint name (under config.savedir)
    # logging
    use_wandb=False,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",  # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb:
    import wandb

    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)


cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from fairseq.tasks.translation import TranslationConfig, TranslationTask


## setup task
task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)


logger.info("loading data for epoch 1")
task.load_dataset(
    split="train", epoch=1, combine=True
)  # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)


sample = task.dataset("valid")[1]
pprint.pprint(sample)
pprint.pprint(
    "Source: "
    + task.source_dictionary.string(
        sample["source"],
        config.post_process,
    )
)
pprint.pprint(
    "Target: "
    + task.target_dictionary.string(
        sample["target"],
        config.post_process,
    )
)


def load_data_iterator(
    task, split, epoch=1, max_tokens=4000, num_workers=0, cached=True
):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond
        # first call of this method has no effect.
    )
    return batch_iterator


demo_epoch_obj = load_data_iterator(
    task, "valid", epoch=1, max_tokens=20, num_workers=0, cached=False
)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
print(sample)


from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
)


class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra


from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)


def build_model(args, task):
    """build a model instance based on hyperparameters"""
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(
        len(src_dict), args.encoder_embed_dim, src_dict.pad()
    )
    decoder_embed_tokens = nn.Embedding(
        len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad()
    )

    # encoder decoder
    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)

    # initialization for seq2seq model is important, requires extra handling
    def init_params(module):
        from fairseq.modules import MultiheadAttention

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)

    # weight initialization
    model.apply(init_params)
    return model


arch_args = Namespace(
    encoder_embed_dim=512,
    encoder_ffn_embed_dim=2048,
    encoder_layers=6,
    decoder_embed_dim=512,
    decoder_ffn_embed_dim=2048,
    decoder_layers=6,
    share_decoder_input_output_embed=True,
    dropout=0.1,
)


def add_transformer_args(args):
    args.encoder_attention_heads = 8
    args.encoder_normalize_before = True

    args.decoder_attention_heads = 8
    args.decoder_normalize_before = True

    args.activation_fn = "relu"
    args.max_source_positions = 1024
    args.max_target_positions = 1024

    # patches on default parameters for Transformer (those not set above)
    from fairseq.models.transformer import base_architecture

    base_architecture(arch_args)


add_transformer_args(arch_args)


if config.use_wandb:
    wandb.config.update(vars(arch_args))


model = build_model(arch_args, task)
logger.info(model)


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: Negative log likelihood，the cross-entropy when target is one-hot. following line is same as F.nll_loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        #  reserve some probability for other labels. thus when calculating cross-entropy,
        # equivalent to summing the log probs of all labels
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # when calculating cross-entropy, add the loss of other labels
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss


# generally, 0.1 is good enough
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (
            0
            if not step
            else self.factor
            * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
            )
        )


optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim,
    factor=config.lr_factor,
    warmup=config.lr_warmup,
    optimizer=torch.optim.AdamW(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001
    ),
)
plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
plt.xlabel("step")
plt.ylabel("lr")
plt.grid(True)
plt.show()


from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(
        itr, accum_steps
    )  # gradient accumulation: update every accum_steps samples

    stats = {"loss": []}
    scaler = GradScaler()  # automatic mixed precision (amp)

    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # gradient accumulation: update every accum_steps samples
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            # mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(
            1 / (sample_size or 1.0)
        )  # (sample_size or 1.0) handles the case of a zero gradient
        gnorm = nn.utils.clip_grad_norm_(
            model.parameters(), config.clip_norm
        )  # grad norm clipping prevents gradient exploding

        scaler.step(optimizer)
        scaler.update()

        # logging
        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log(
                {
                    "train/loss": loss_print,
                    "train/grad_norm": gnorm.item(),
                    "train/lr": optimizer.rate(),
                    "train/sample_size": sample_size,
                }
            )

    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats


# fairseq's beam search generator
# given model and input seqeunce, produce translation hypotheses by beam search
sequence_generator = task.build_generator([model], config)


def decode(toks, dictionary):
    # convert from Tensor to human readable sentence
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"


def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
        srcs.append(
            decode(
                utils.strip_pad(
                    sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()
                ),
                task.source_dictionary,
            )
        )
        hyps.append(
            decode(
                gen_out[i][0]["tokens"],  # 0 indicates using the top hypothesis in beam
                task.target_dictionary,
            )
        )
        refs.append(
            decode(
                utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
                task.target_dictionary,
            )
        )
    return srcs, hyps, refs


import shutil
import sacrebleu


def validate(model, task, criterion, log_to_wandb=True):
    logger.info("begin validation")
    itr = load_data_iterator(
        task, "valid", 1, config.max_tokens, config.num_workers
    ).next_epoch_itr(shuffle=False)

    stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
    srcs = []
    hyps = []
    refs = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = (
                criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                / sample_size
            )
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)

            # do inference
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)

    tok = "zh" if task.cfg.target_lang == "zh" else "13a"
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)  # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs

    if config.use_wandb and log_to_wandb:
        wandb.log(
            {
                "valid/loss": stats["loss"],
                "valid/bleu": stats["bleu"].score,
            },
            commit=False,
        )

    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])

    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats


def validate_and_save(model, task, criterion, optimizer, epoch, save=True):
    stats = validate(model, task, criterion)
    bleu = stats["bleu"]
    loss = stats["loss"]
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)

        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step},
        }
        torch.save(check, savedir / f"checkpoint{epoch}.pt")
        shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

        # save epoch samples
        with open(
            savedir / f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir / f"checkpoint_best.pt")

        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

    return stats


def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir) / name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(
            f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}"
        )
    else:
        logger.info(f"no checkpoints found at {checkpath}!")


model = model.to(device=device)
criterion = criterion.to(device=device)


logger.info("task: {}".format(task.__class__.__name__))
logger.info("encoder: {}".format(model.encoder.__class__.__name__))
logger.info("decoder: {}".format(model.decoder.__class__.__name__))
logger.info("criterion: {}".format(criterion.__class__.__name__))
logger.info("optimizer: {}".format(optimizer.__class__.__name__))
logger.info(
    "num. model params: {:,} (num. trained: {:,})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
)
logger.info(
    f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}"
)


epoch_itr = load_data_iterator(
    task, "train", config.start_epoch, config.max_tokens, config.num_workers
)
try_load_checkpoint(model, optimizer, name=config.resume)
while epoch_itr.next_epoch_idx <= config.max_epoch:
    # train for one epoch
    train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
    stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
    logger.info("end of epoch {}".format(epoch_itr.epoch))
    epoch_itr = load_data_iterator(
        task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers
    )


checkdir = config.savedir
command = [
    "python",
    "./fairseq/scripts/average_checkpoints.py",
    "--inputs",
    checkdir,
    "--num-epoch-checkpoints",
    "5",
    "--output",
    f"{checkdir}/avg_last_5_checkpoint.pt",
]
subprocess.run(command, shell=True)


# checkpoint_last.pt : latest epoch
# checkpoint_best.pt : highest validation bleu
# avg_last_5_checkpoint.pt:　the average of last 5 epochs
try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
validate(model, task, criterion, log_to_wandb=False)


def generate_prediction(model, task, split="test", outfile="./prediction.txt"):
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(
        task, split, 1, config.max_tokens, config.num_workers
    ).next_epoch_itr(shuffle=False)

    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)

            # do inference
            s, h, r = inference_step(sample, model)

            hyps.extend(h)
            idxs.extend(list(sample["id"]))

    # sort based on the order before preprocess
    hyps = [x for _, x in sorted(zip(idxs, hyps))]

    with open(outfile, "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h + "\n")


mono_dataset_name = "mono"
mono_prefix = Path(data_dir).absolute() / mono_dataset_name
mono_data = f"{mono_prefix}/mono.raw.zh"
min_len = 1
max_len = 1000
cleaned_mono_data = f"{mono_prefix}/mono.raw.clean.zh"
if Path(cleaned_mono_data).exists():
    print(f"{cleaned_mono_data} exists. skipping clean.")
else:
    with open(mono_data, "r", encoding="utf-8") as in_f:
        with open(cleaned_mono_data, "w", encoding="utf-8") as out_f:
            for line in in_f:
                line = line.strip()
                cleaned_line = clean_s(line, "zh")
                line_len = len_s(cleaned_line, "zh")
                if min_len > 0 and line_len < min_len:
                    continue
                if max_len > 0 and line_len > max_len:
                    continue
                print(cleaned_line, file=out_f)


mono_zh = mono_prefix / "mono.zh"
mono_en = mono_prefix / "mono.en"
if mono_zh.exists():
    print(f"{mono_zh} exists. skipping spm_encode.")
else:
    with open(cleaned_mono_data, "r", encoding="utf-8") as in_f:
        with open(mono_zh, "w", encoding="utf-8") as out_zh:
            with open(mono_en, "w", encoding="utf-8") as out_en:
                for line in in_f:
                    line = line.strip()
                    tok = spm_model.encode(line, out_type=str)
                    print(" ".join(tok), file=out_zh)
                    print(".", file=out_en)


binpath_mono = Path("./DATA/data-bin", "mono")
src_dict_file = "./DATA/data-bin/ted2020/dict.zh.txt"
tgt_dict_file = "./DATA/data-bin/ted2020/dict.en.txt"
if binpath_mono.exists():
    print(binpath_mono, "exists, will not overwrite!")
else:
    cmd = [
        "python",
        "-m",
        "fairseq_cli.preprocess",
        "--source-lang",
        "zh",
        "--target-lang",
        "en",
        "--trainpref",
        str(mono_prefix / "mono"),
        "--destdir",
        str(binpath_mono),
        "--srcdict",
        src_dict_file,
        "--tgtdict",
        tgt_dict_file,
        "--workers",
        "2",
    ]
    subprocess.run(cmd, check=True)


files_to_copy = [
    (binpath_mono / "train.zh-en.zh.bin", binpath / "mono.zh-en.zh.bin"),
    (binpath_mono / "train.zh-en.zh.idx", binpath / "mono.zh-en.zh.idx"),
    (binpath_mono / "train.zh-en.en.bin", binpath / "mono.zh-en.en.bin"),
    (binpath_mono / "train.zh-en.en.idx", binpath / "mono.zh-en.en.idx"),
]
for src, dst in files_to_copy:
    dst = Path(dst)
    if not dst.exists():
        shutil.copy2(src, dst)

back_pred = Path("./", "back_prediction.en")
if not back_pred.exists():
    generate_prediction(model, task, split="mono", outfile=back_pred)


with open(back_pred, "r", encoding="utf-8") as in_f:
    with open(mono_en, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            tok = spm_model.encode(line, out_type=str)
            print(" ".join(tok), file=out_f)


binpath_synthetic = Path("./DATA/data-bin", "synthetic")
src_dict_file = "./DATA/data-bin/ted2020/dict.en.txt"
tgt_dict_file = "./DATA/data-bin/ted2020/dict.zh.txt"
if binpath_synthetic.exists():
    print(binpath_synthetic, "exists, will not overwrite!")
else:
    cmd = [
        "python",
        "-m",
        "fairseq_cli.preprocess",
        "--source-lang",
        "en",
        "--target-lang",
        "zh",
        "--trainpref",
        str(mono_prefix / "mono"),
        "--destdir",
        str(binpath_synthetic),
        "--srcdict",
        src_dict_file,
        "--tgtdict",
        tgt_dict_file,
        "--workers",
        "2",
    ]
    subprocess.run(cmd, check=True)


binpath_ted_with_mono = Path("./DATA/data-bin", "ted2020_with_mono")
if not binpath_ted_with_mono.exists():
    shutil.copytree(binpath, binpath_ted_with_mono, dirs_exist_ok=True)


files_to_copy = [
    (
        binpath_synthetic / "train.en-zh.en.bin",
        binpath_ted_with_mono / "train1.en-zh.en.bin",
    ),
    (
        binpath_synthetic / "train.en-zh.en.idx",
        binpath_ted_with_mono / "train1.en-zh.en.idx",
    ),
    (
        binpath_synthetic / "train.en-zh.zh.bin",
        binpath_ted_with_mono / "train1.en-zh.zh.bin",
    ),
    (
        binpath_synthetic / "train.en-zh.zh.idx",
        binpath_ted_with_mono / "train1.en-zh.zh.idx",
    ),
]
for src, dst in files_to_copy:
    dst = Path(dst)
    if not dst.exists():
        shutil.copy2(src, dst)
