from asyncio.log import logger
import pickle, os
from time import time
from tqdm import tqdm
import torch
from utils import set_random_seed
from torch import nn
from torch.optim import Adam, lr_scheduler
from transformers.optimization import Adafactor, AdafactorSchedule, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoModel
from torch.utils.data import SequentialSampler, DataLoader
from model import EVAModel, EVATokenizer
from dataset import EVADataset
from samplers import RandomSampler, DistributedBatchSampler
from eva_evaluate import evaluate
import utils
from generation_metrics import Metric
from arguments import init_argument
from torch.utils.tensorboard import SummaryWriter   


# 获取数据
def get_datasets(args, data_type, tokenizer, train_logger):

    cache_path = args.cache_path + "_no_knowledge"
    My_Dataset = EVADataset

    dataset_path = os.path.join(args.data_path, data_type + "_dialog.txt")  # **/data/kdconv/train_dialog.txt

    dataset = My_Dataset(
        args,
        tokenizer,
        dialog_path=dataset_path,
        split=data_type,
        ratio=args.ratio,
        cache_path=cache_path)

    train_logger.info("加载 {} {} 数据集, Dataset_lengrh:{}, max_enc_len: {}, max_dec_len: {}, Max kg enc len: {}".format(args.data_domin, 
                                                                                                        data_type, 
                                                                                                        dataset.__len__, 
                                                                                                        args.enc_seq_length, args.dec_seq_length,
                                                                                                        args.enc_kg_length))


    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             drop_last=True,
                             collate_fn=dataset.collate)

    return data_loader

# 在没有finetune之前进行评测
def no_train_test(args, model, tokenizer, test_dataloader, device):
    eval_loss, test_metric_res, _ = evaluate(args, tokenizer, eval_data_loader=test_dataloader, model=model,
                                            device=device)
    eval_log_string = ""
    for key, value in test_metric_res.items():
        eval_log_string += " {}: {:.5} | ".format(key, value)

    return eval_loss, eval_log_string

# 在没有finetune之后进行评测
def after_train_test(args, tokenizer, test_dataloader, device):
    # 1.0 获得文件夹下面所有的训练好的模型
    train_model_save_path = os.path.join(args.save_finetune, args.save_model_path)
    model_list = os.listdir(train_model_save_path)

    for model_name in model_list:
        if "epoch" not in model_name:
            continue
        model = torch.load(os.path.join(train_model_save_path, model_name))
        model = model.to(device)

        _, test_metric_res, generation_res = evaluate(args, tokenizer, eval_data_loader=test_dataloader, model=model,
                                                device=device)
            
        train_logger.warning("==============test metrics: model {}'s metric_res==========".format(model_name))
        utils.write_metric_to_log(test_metric_res, train_logger)

        utils.write_gens_to_file(model_name, generation_res, args, "test")

# 训练
def train(args, model, tokenizer, optimizer, scheduler, train_dataloader, valid_dataloader, test_dataloader, device):
    train_step, steps_loss, lowest_valid_loss = 0, 0.0, 100.0
    # 最佳指标dict
    best_metric_dict = {"bleu-4":[0.0, 0.0, 0.0]}
    valid_early_stop_num, train_loss_early_stop_epoch, lowest_loss = 0, 0, 0.0
    # train model存放位置 
    train_model_save_path = os.path.join(args.save_finetune, args.save_model_path)
    if not os.path.exists(train_model_save_path):
        os.mkdir(train_model_save_path)

    for epoch in range(args.train_from_middle_epoch, args.epochs):
        model.train()
        for i, model_batch in enumerate(tqdm(train_dataloader, desc="Training")):
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)
            train_step += 1
            output = model(**model_batch)
            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            steps_loss += loss.item()
            optimizer.step()
            scheduler.step()

            # 第一步，输出loss
            if epoch==args.train_from_middle_epoch and i==0:
                train_logger.info("train start loss:{}".format(loss.item()))
                lowest_loss = loss.item()

            # 进行一次step loss输出
            if train_step % args.loss_step_num == 0:
                current_steps_all_loss, steps_loss = steps_loss/args.loss_step_num, 0.0
                train_logger.warning(f"epoch:{epoch}, step:{train_step}, steps train loss:{current_steps_all_loss}, lr:{optimizer.param_groups[0]['lr']}")
                # logger 播报最新lowest_loss
                if lowest_loss > current_steps_all_loss:
                    train_logger.warning(f"Get new lowest loss... {current_steps_all_loss}")
                # 更新train_loss_early_stop_epoch 和 lowest_loss
                train_loss_early_stop_epoch = epoch if lowest_loss > current_steps_all_loss else train_loss_early_stop_epoch
                lowest_loss = current_steps_all_loss if lowest_loss > current_steps_all_loss else lowest_loss


            # 验证 valid
            if args.do_valid and train_step % args.valid_step_num == 0:
                valid_loss, valid_metric_res, generation_res = evaluate(args, tokenizer, eval_data_loader=valid_dataloader, model=model, device=device)
                model.train()  # 回到model.train()
                train_logger.warning(f"steps valid loss:{valid_loss}")
                if round(valid_loss, 5) > lowest_valid_loss:
                    lowest_valid_loss = round(valid_loss, 5)
                    torch.save(model, f"{train_model_save_path}/lowest_valid_loss_model.pt")
                    train_logger.warning(f"Save lowest valid_loss model, valid loss: {lowest_valid_loss}")
                # train_logger.warning("=============={}/{} epoch/epochs {}/{} step/steps valid's metric_res==========".format(epoch, args.epochs, train_step, args.train_steps))
                if generation_res:
                    # 将valid metric写入log
                    utils.write_metric_to_log(valid_metric_res, train_logger)
                    # 保存模型
                    best_metric_dict, save_new = utils.save_model_while_train(train_model_save_path, valid_metric_res, 
                                                                            best_metric_dict, model, epoch, train_step, train_logger, args)
                    valid_early_stop_num = valid_early_stop_num + 1 if not save_new else 0
                    # 如果保存了新模型，将valid generation写入txt
                    if save_new:
                        utils.write_gens_to_file(epoch, generation_res, args, "valid")



if __name__ == '__main__':
    # 获取所有参数
    args = init_argument()

    train_logger = utils.get_logger(args.train_log_file)

    # 设置device
    device = 'cuda:{}'.format(args.gpu_index) if torch.cuda.is_available() else 'cpu'
    set_random_seed(args.seed)  # 设置随机种子
    
    # train日志中输入所有超参数
    train_logger.critical("train is start...\n device: {}".format(device))
    train_logger.info(args)
    utils.write_args_to_logger(args, train_logger)

    # 加载tokenizer，导入数据
    tokenizer = EVATokenizer.from_pretrained(args.tokenizer_path)
    train_dataloader = get_datasets(args, "train", tokenizer, train_logger)

    valid_dataloader = get_datasets(args, 'valid', tokenizer, train_logger)
    test_dataloader = get_datasets(args, 'test', tokenizer,train_logger)
    args.train_steps = len(train_dataloader) * args.epochs

    # 加载模型
    train_logger.info("加载model......")
    model = utils.get_model(args)
    model = model.to(device)


    # 加载优化器，调度器
    optimizer, scheduler = utils.get_optimizer_scheduler(args, model)

    train(args, model, tokenizer, optimizer, scheduler, train_dataloader, valid_dataloader, test_dataloader, device)

    train_logger.info("end...")


    # finetune前
    # msg = "|有|外部知识," if args.if_kg else "|无|外部知识" 
    # _, eval_log_string = no_train_test(args, model, tokenizer, valid_dataloader, device)
    # print(msg + "\n tune之前的评测结果: \n" + eval_log_string)
    # train_logger.info(msg + "\n tune之前的评测结果: \n" + eval_log_string)

    # finetune后
    # after_train_test(args, tokenizer, test_dataloader, device)

