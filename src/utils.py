# coding=utf-8

"""Utilities for logging and serialization"""

import random, os
import numpy as np
import torch
import logging
from torch.optim import Adam
from transformers.optimization import Adafactor, get_linear_schedule_with_warmup
from model import EVAModel, EVATokenizer


def save_rank_0(args, message):
    with open(args.log_file, "a") as f:
        f.write(message + "\n")
        f.flush()

def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_logger2(log_name):
    logging.basicConfig(level=logging.DEBUG,
                    filename=log_name,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def get_logger(name):
    log_name = name[:-4].split("/")[-1]
    logger = logging.getLogger(log_name)
    # 创建一个handler，用于写入日志文件
    filename = name
    fh = logging.FileHandler(filename, mode='a+', encoding='utf-8')
    # 定义输出格式(可以定义多个输出格式例formatter1，formatter2)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    # 定义日志输出层级
    logger.setLevel(logging.DEBUG)
    # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
    fh.setFormatter(formatter)
    # 给logger对象绑定文件操作符
    logger.addHandler(fh)
    return logger

def write_args_to_logger(args, logger):
    logger.info("model-size: {}".format(args.model_size))
    logger.info("if-kg: {}".format(args.if_kg))
    logger.info("epochs: {}".format(args.epochs))
    logger.info("data-domin: {}".format(args.data_domin))
    logger.info("batch-size: {}".format(args.batch_size))
    logger.info("gpu-index: {}".format(args.gpu_index))
    logger.info("data-path: {}".format(args.data_path))
    logger.info("cache-path: {}".format(args.cache_path))
    logger.info("log-file: {}".format(args.train_log_file))
    logger.info("save-finetune-path: {}".format(args.save_finetune))
    logger.info("save-model-path: {}".format(args.save_model_path))
    logger.info("lr: {}".format(args.lr))
    logger.info("train-steps: {}".format(args.train_steps))
    logger.info("valid-step-num: {}".format(args.valid_step_num))
    logger.info("warmup: {}".format(args.warmup))

def write_gen_args_to_logger(args, logger):
    logger.info("model-path: {}".format(args.pretrain_model_path))
    logger.info("model-size: {}".format(args.model_size))
    logger.info("if-prefix: {}".format(args.prefix))
    logger.info("if-kg: {}".format(args.if_kg))
    logger.info("data-domin: {}".format(args.data_domin))
    logger.info("batch-size: {}".format(args.batch_size))
    logger.info("以下为生成超参数.....")
    logger.info("do-sample: {}".format(args.do_sample))
    logger.info("temperature: {}".format(args.temperature))
    logger.info("top-p: {}".format(args.top_p))
    logger.info("top-k: {}".format(args.top_k))
    logger.info("max-generation-length: {}".format(args.max_generation_length))
    logger.info("min-generation-length: {}".format(args.min_generation_length))
    logger.info("num-beams: {}".format(args.num_beams))
    logger.info("no-repeat-ngram-size: {}".format(args.no_repeat_ngram_size))
    logger.info("repetition-penalty: {}".format(args.repetition_penalty))
    logger.info("length-penalty: {}".format(args.length_penalty))


def save_model_while_train(train_model_save_path, valid_metric_res, best_metric_dict, model, epoch, step, train_logger, args):
    save_new = True # 该阶段是否保存了新的指标

    if valid_metric_res["bleu-4"] > best_metric_dict["bleu-4"][0] or valid_metric_res["bleu-4"] > best_metric_dict["bleu-4"][1] or valid_metric_res["bleu-4"] > best_metric_dict["bleu-4"][2]:
        torch.save(model, os.path.join(train_model_save_path, "from_{}_epoch_{}-step_{}-bleu4_{:.5}.pt".format(args.train_from_middle_epoch, epoch, step, valid_metric_res["bleu-4"])))
        best_metric_dict["bleu-4"].sort()  # 从小到大排序
        best_metric_dict["bleu-4"][0] = valid_metric_res["bleu-4"]
        train_logger.warning("Save model... model name: {}".format(os.path.join(train_model_save_path, 
                                                            "from_{}_epoch_{}-step_{}-bleu4_{:.5}.pt".format(args.train_from_middle_epoch, epoch, step, valid_metric_res["bleu-4"]))))
        train_logger.warning("Now best_metric_dict[\"bleu-4\"]: {}".format(best_metric_dict["bleu-4"]))

        # 清除不需要的模型
        for file_name in os.listdir(train_model_save_path):
            if "bleu4" in file_name  and "from_{}".format(args.train_from_middle_epoch) in file_name \
                                        and "bleu4_{:.5}.pt".format(best_metric_dict["bleu-4"][0]) not in file_name \
                                        and "bleu4_{:.5}.pt".format(best_metric_dict["bleu-4"][1]) not in file_name \
                                            and "bleu4_{:.5}.pt".format(best_metric_dict["bleu-4"][2]) not in file_name:
                os.remove(os.path.join(train_model_save_path, file_name))
                train_logger.warning("Delete model... model name: {}".format(os.path.join(train_model_save_path, file_name)))
    else:
        save_new = False

    return best_metric_dict, save_new


def save_state_while_train(train_model_save_path, epoch, model, optimizer, scheduler, train_logger):
    save_path = os.path.join(train_model_save_path, f"{epoch}_state")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 依次保存模型, 优化器, 调度器
    torch.save(model, os.path.join(save_path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
    train_logger.warning(f"{epoch}阶段 model, optimizer, scheduler 保存成功, 保存位置{save_path}")


def write_metric_to_log(test_metric_res, train_logger):
    others_log_string = ""
    dist_log_string = ""
    bleu_log_string = ""
    for key, value in test_metric_res.items():
        if "dist" in key:
            dist_log_string += " {}: {:.5} | ".format(key, value)
        elif "bleu" in key:
            bleu_log_string += " {}: {:.5} | ".format(key, value)
        else:
            others_log_string += " {}: {:.5} | ".format(key, value)
    train_logger.info(dist_log_string)
    train_logger.info(bleu_log_string)
    train_logger.info(others_log_string)

def write_gens_to_file(model_name, generation_res, args, term):
    gens_path = os.path.join(args.save_finetune, args.save_model_path)
    gens_file = os.path.join(gens_path, "{}_generation.txt".format(term))

    if not os.path.exists(gens_file):
        f = open(gens_file, "w")
    else:
        f = open(gens_file, "a")
    
    if term == "test":
        f.write("{} model gengration examples....\n".format(model_name))
    else:
        f.write("{} epoch gengration examples....\n".format(model_name))

    for gen in generation_res[:12]:
        f.write("****** context:{} \n ****** response:{} \n ****** generation:{} \n ".format(gen["context"], gen["response"], gen["generation"]))
        f.write("---------------------------------------------------------------------------------------\n")

    f.close()


def get_optimizer_scheduler(args, model):

    # 设置优化器
    if not args.adafactor:
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        # 这只Adafactor优化器
        optimizer = Adafactor(model.parameters(), weight_decay=args.weight_decay, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup * args.train_steps, num_training_steps=args.train_steps)

    if args.train_from_middle_epoch != 0:
        train_model_save_path = os.path.join(args.save_finetune, args.save_model_path)
        optimizer.load_state_dict(torch.load(f"{train_model_save_path}/{args.train_from_middle_epoch}_state/optimizer.pt"))
        scheduler.load_state_dict(torch.load(f"{train_model_save_path}/{args.train_from_middle_epoch}_state/scheduler.pt"))

    return optimizer, scheduler

def get_model(args):
    if args.train_from_middle_epoch != 0:
        train_model_save_path = os.path.join(args.save_finetune, args.save_model_path)
        model = torch.load(f"{train_model_save_path}/{args.train_from_middle_epoch}_state/model.pt")

        return model
    
    model = EVAModel.from_pretrained(args.pretrain_model_path)

    return model