# 对话生成参数
import argparse
import os


def add_text_generate_args(parser: argparse.ArgumentParser):
    """Text generate arguments."""

    group = parser.add_argument_group("Text generation", "configurations")
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--temperature", type=float, default=0.9,
                       help="The temperature of sampling.")
    group.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p sampling.")
    group.add_argument("--top-k", type=int, default=0,
                       help="Top-k sampling.")
    group.add_argument("--max-generation-length", type=int, default=30,
                       help="The maximum sequence length to generate.")
    group.add_argument("--min-generation-length", type=int, default=2,
                       help="The minimum sequence length to generate.")
    group.add_argument("--num-beams", type=int, default=4,
                       help="The beam number of beam search.")
    group.add_argument("--no-repeat-ngram-size", type=int, default=4,
                       help="The n_gram whose length is less than this option will appear at most once in the whole dialog.")
    group.add_argument("--repetition-penalty", type=float, default=1.6,
                       help="Repetition penalty, to prevent repeated words.")
    group.add_argument("--length-penalty", type=float, default=1.6,
                       help="Length penalty, to prevent short generation.")
    return parser

# 训练过程参数
def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group("train", "training configurations")
    group.add_argument("--train_from_middle_epoch", default=4, type=int, 
                choices=[0, 4, 24, 44, 64, 84], 
                help="从中间哪个epoch开始训练.")
    group.add_argument("--seed", type=int, default=422, help="Set the random seed.")
    group.add_argument('--epochs', default=20, type=int, help='')
    group.add_argument('--train-iters', default=-1, help='')
    group.add_argument('--train-ratio', default=1, help='')
    group.add_argument('--train-steps', default=100000, type=int, help='')
    group.add_argument('--loss-step-num', default=60, type=int, help='')
    group.add_argument('--valid-step-num', default=200, type=int, help='')
    group.add_argument('--valid_early_stop_num', default=10, type=int, help='')
    group.add_argument('--train_loss_early_stop_epoch', default=3, type=int, help='')
    # Learning rate.
    group.add_argument("--adafactor", action="store_true", help="if use adafactor optimizer.")
    group.add_argument("--lr", type=float, default=2.0e-4, help="Initial learning rate.")
    group.add_argument("--lr-decay-style", type=str, default="linear",
                       choices=["constant", "linear", "cosine", "exponential", "noam"],
                       help="Learning rate decay function.")
    group.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay coefficient for L2 regularization.")
    group.add_argument("--warmup", type=float, default=0.03,
                       help="Percentage of training steps for warmup.")
    # train valid eval
    group.add_argument("--do-train", action="store_true",
                       help="Do model training.")
    group.add_argument("--do-valid", action="store_true",
                       help="Do model validation while training.")
    group.add_argument("--do-eval",  action="store_true",
                       help="Do model evaluation/inference.")
    group.add_argument("--eval-generation", action="store_true",
                       help="Maximum encoder sequence length to process.")
    return parser


# 模型参数
def add_model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("model", "model configurations")
    group.add_argument('--pretrain-model-path', default="", help="预训练模型位置")
    group.add_argument('--model-size', default="base", type=str, help="模型规模")
    group.add_argument('--gpu-index', default=0, type=int, help="使用哪块GPU")
    group.add_argument('--model-config', default="", help="预训练模型config文件位置")
    group.add_argument('--tokenizer-path', default="", help="tokenize path")
    # 训练过程中模型存储位置
    group.add_argument('--save-finetune', default="", help="finetune文件存储位置")
    group.add_argument('--save-model-path', default="", help="finetune 保存的模型文件夹")
    return parser

# 数据参数
def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Data configs", "configurations")
    group.add_argument('--data-path', type=str,default="")
    group.add_argument('--cache-path', default="", help='')
    group.add_argument('--data-domin', default="film", help="语料的种类")
    group.add_argument('--enc-seq-length', type=int, default=128, help='')
    group.add_argument('--dec-seq-length', type=int, default=128, help='')
    group.add_argument('--enc-kg-length', type=int, default=512, help='知识长度')
    group.add_argument("--if-kg", action="store_true", help="if use fine-tuning with Knowledge.")
    group.add_argument('--num-workers', default=2, type=int, help='')
    group.add_argument('--drop-last', default=True, help='')
    group.add_argument('--fp16', action="store_true", help='')
    group.add_argument('--batch-size', default=16, type=int, help='')
    group.add_argument('--ratio', default=1.0, type=float, help='')

    return parser

#log参数
def add_log_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Data configs", "configurations")
    group.add_argument('--train-logs-path', default="")
    group.add_argument('--train-log-file', default="")
    group.add_argument('--eval-log-file', default="")
    group.add_argument('--tensorboard-path', default="")

    return parser
    
# 获取全部参数
def init_argument():
    parser = argparse.ArgumentParser()

    parser = add_data_args(parser)
    parser = add_training_args(parser)
    parser = add_model_args(parser)
    parser = add_text_generate_args(parser)
    parser = add_log_args(parser)
    args = parser.parse_args()

    return args