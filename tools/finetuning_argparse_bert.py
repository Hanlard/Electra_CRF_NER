import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters


    parser.add_argument("--task_name", default="cner", type=str,
                        help="The name of the task to train selected in the list: ")

    ##数据集
    parser.add_argument("--data_dir", default="datasets/35wnltk&js", type=str,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: [bert,albert,albert_lstm]")

    ## 预训练模型路径&名称
    parser.add_argument("--model_name_or_path", default="prev_trained_model/bert_zh_base", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " )

    parser.add_argument("--optimizer", default="AdamW", type=str,
                        help="optimizer:AdamW/Lamb" )

    ## 模型保存及预测结果
    parser.add_argument("--output_dir", default="outputs/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    parser.add_argument("--config_name", default="prev_trained_model/bert_zh_base/config.json", type=str,
                        help="【配置文件】Pretrained config name or path if not the same as model_name")

    parser.add_argument("--tokenizer_name", default="prev_trained_model/bert_zh_base/vocab.txt", type=str,
                        help="【词表路径vocab.txt】Pretrained tokenizer name or path if not the same as model_name", )

    parser.add_argument("--contiu_train", default="", type=str,
                        help="从上一步保存开始训练", )



    # Other parameters

    parser.add_argument('--fix', default=False, type=bool, help="是否使用后期修复")

    parser.add_argument('--markup', default='bio', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true", default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", default=False,
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",default="True",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    # adversarial training
    parser.add_argument("--do_adv", action="store_true",
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints",type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    return parser