# coding=utf-8

"""Evaluate EVA"""

import os
import json
import torch
from torch.utils.data import DataLoader, DistributedSampler
from model import EVATokenizer

from utils import print_args, save_rank_0
from utils import set_random_seed
from model import EVAModel

from generation_metrics import Metric
from dataset.eva_datasets import EVADataset
from tqdm import tqdm


def load_data(args, data_type, tokenizer, ratio=1, drop_last=True):
    data_path = os.path.join(args.data_path, data_type + args.data_ext)

    # Data parallel arguments.
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    num_workers = args.num_workers

    dataset = EVADataset(
        args,
        tokenizer,
        data_path,
        data_type,
        ratio=ratio,
        cache_path=args.cache_path)

    batch_sampler = DistributedSampler(dataset, num_replicas=int(os.getenv("WORLD_SIZE", "1")),
                                       rank=int(os.getenv("RANK", "0")), shuffle=False, drop_last=drop_last)

    data_loader = DataLoader(dataset,
                             batch_size=args.eval_batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False,
                             drop_last=drop_last,
                             collate_fn=dataset.collate)

    # Torch dataloader.
    return data_loader, dataset, batch_sampler


def evaluate(args, tokenizer, eval_data_loader, model: EVAModel, device):
    """Evaluation."""

    model.eval()

    metric = Metric(tokenizer)

    generation_res = None
    metric_res = {}
    generation_res = []

    step = 0

    with torch.no_grad():
        loss_res = 0.0
        for batch in tqdm(eval_data_loader, desc="Computing Loss"):
            for k in batch:
                batch[k] = batch[k].to(device)
            forw_out = model(**batch)
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            loss_res += loss
            step += 1

        loss_res /= step

        print(loss_res)

        if args.eval_generation:
            generation_res = []
            for e, batch in enumerate(tqdm(eval_data_loader, desc="Evaluating")):
                model_gen_tokens = model.generate(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    max_length=args.max_generation_length,
                    min_length=args.min_generation_length,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    length_penalty=args.length_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    encoder_no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                    use_cache=True
                )

                model_gen_str = tokenizer.batch_decode(model_gen_tokens, skip_special_tokens=True)
                label_str = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                context_str = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

                for lab, gen in zip(label_str, model_gen_str):
                    metric.forstr([lab], gen)

                for ctx, lab, gen in zip(context_str, label_str, model_gen_str):
                    generation_res.append({
                        'context': ctx,
                        'response': lab,
                        'generation': gen,
                    })
                    if e == 0:
                        print(f'****** context: {ctx}\n'
                              f'****** response: {lab}\n'
                              f'****** generation: {gen}\n')
    

            metric_res, *_ = metric.close()

        metric_res["loss"] = loss_res

    return loss_res, metric_res, generation_res
