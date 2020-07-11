import os
import json
import random
import numpy as np
import collections
from pathlib import Path
from tools.common import logger, init_logger
from argparse import ArgumentParser
from tools.common import seed_everything
from model.tokenization_bert import BertTokenizer
from callback.progressbar import ProgressBar

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words):
    """Creates `TrainingInstance`s for a single document.
     This method is changed to create sentence-order prediction (SOP) followed by idea from paper of ALBERT, 2019-08-28, brightmart
    """
    document = all_documents[document_index]  # �õ�һ���ĵ�

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:  # ��һ���ı�������10%�ĸ��ʣ�����ʹ�ñȽ϶̵����г��ȣ��Ի���Ԥѵ���ĳ����к͵��Ž׶Σ����ܵģ������еĲ�һ�����
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    # �跨ʹ��ʵ�ʵľ��ӣ�����������ĽضϾ��ӣ��Ӷ����õĹ������������Ԥ�������
    instances = []
    current_chunk = []  # ��ǰ������ı��Σ������������
    current_length = 0
    i = 0
    while i < len(document):  # ���ĵ��ĵ�һ��λ�ÿ�ʼ���������¿�
        segment = document[
            i]  # segment���б�������ǰ��ַֿ���һ���������ӣ��� segment=['��', '��', 'һ', 'ү', '��', '��', '��', '��', '��', '��', '��', '��', 'ϱ', '��', '��', '��', '��', '��', '��', '��']
        # segment = get_new_segment(segment)  # whole word mask for chinese: ��Ϸִʵ����ĵ�whole mask���ü�����Ҫ�ĵط����ϡ�##��
        current_chunk.append(segment)  # ��һ�������ľ��Ӽ��뵽��ǰ���ı�����
        current_length += len(segment)  # �ۼƵ�Ϊֹλ�ýӴ������ӵ��ܳ���
        if i == len(document) - 1 or current_length >= target_seq_length:
            # ����ۼƵ����г��ȴﵽ��Ŀ��ĳ��ȣ���ǰ�ߵ����ĵ���β==>���첢��ӵ���A[SEP]B���е�A��B�У�
            if current_chunk:  # �����ǰ�鲻Ϊ��
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:  # ��ǰ�飬������������������ӣ�ȡ��ǰ���һ������Ϊ��A[SEP]B���е�A����
                    a_end = random.randint(1, len(current_chunk) - 1)
                # ����ǰ�ı�����ѡȡ������ǰ�벿�֣���ֵ��A��tokens_a
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # ���조A[SEP]B���е�B����(��һ�����������ĵ�ǰ�ĵ��еĺ�벿;��ԭBERT��ʵ����һ����������Ĵ���һ���ĵ���ѡȡ�ģ���
                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                # �аٷ�֮50%�ĸ��ʽ���һ��tokens_a��tokens_b��λ��
                # print("tokens_a length1:",len(tokens_a))
                # print("tokens_b length1:",len(tokens_b)) # len(tokens_b) = 0
                if len(tokens_a) == 0 or len(tokens_b) == 0: continue
                if random.random() < 0.5:  # ����һ��tokens_a��tokens_b
                    is_random_next = True
                    temp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = temp
                else:
                    is_random_next = False
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # ��tokens_a & tokens_b���뵽����bert�ķ�񣬼���[CLS]tokens_a[SEP]tokens_b[SEP]����ʽ����ϵ�һ����Ϊ���յ�tokens; Ҳ����segment_ids��ǰ�沿��segment_ids��ֵ��0�����沿�ֵ�ֵ��1.
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
                original_tokens = tokens.copy()
                # ����masked LM����������� Creates the predictions for the masked LM objective
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words)
                instance = {
                    'original_tokens':original_tokens,
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels}
                instances.append(instance)
            current_chunk = []  # ��յ�ǰ��
            current_length = 0  # ���õ�ǰ�ı���ĳ���
        i += 1  # �����ĵ��е���������
    return instances


def create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    # n-gram masking Albert
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indices.append(i)
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_token_labels = []
    covered_indices = set()
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)
        if len(masked_token_labels) >= num_to_mask:
            break
        if index in covered_indices:
            continue
        if index < len(cand_indices) - (n - 1):
            for i in range(n):
                ind = index + i
                if ind in covered_indices:
                    continue
                covered_indices.add(ind)
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[ind]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = random.choice(vocab_list)
                masked_token_labels.append(MaskedLmInstance(index=ind, label=tokens[ind]))
                tokens[ind] = masked_token

    #assert len(masked_token_labels) <= num_to_mask
    masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_token_labels]
    masked_labels = [p.label for p in masked_token_labels]
    return tokens, mask_indices, masked_labels

def create_training_instances(input_file, tokenizer, max_seq_len, short_seq_prob,
                              max_ngram, masked_lm_prob, max_predictions_per_seq):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    f = open(input_file, 'r',encoding="utf-8")
    lines = f.readlines()
    pbar = ProgressBar(n_total=len(lines), desc='read data')
    for line_cnt, line in enumerate(lines):
        line = line.strip()
        # Empty lines are used as document delimiters
        if not line:
            all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
            all_documents[-1].append(tokens)
        pbar(step=line_cnt)
    print(' ')
    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    random.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    pbar = ProgressBar(n_total=len(all_documents), desc='create instances')
    for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_len, short_seq_prob,
                max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words))
        pbar(step=document_index)
    print(' ')
    ex_idx = 0
    while ex_idx < 5:
        instance = instances[ex_idx]
        logger.info("-------------------------Example-----------------------")
        logger.info(f"id: {ex_idx}")
        logger.info(f"original tokens: {' '.join([str(x) for x in instance['original_tokens']])}")
        logger.info(f"tokens: {' '.join([str(x) for x in instance['tokens']])}")
        logger.info(f"masked_lm_labels: {' '.join([str(x) for x in instance['masked_lm_labels']])}")
        logger.info(f"segment_ids: {' '.join([str(x) for x in instance['segment_ids']])}")
        logger.info(f"masked_lm_positions: {' '.join([str(x) for x in instance['masked_lm_positions']])}")
        logger.info(f"is_random_next : {instance['is_random_next']}")
        ex_idx += 1
    random.shuffle(instances)
    return instances


def main():
    parser = ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default="dataset", type=str)
    parser.add_argument("--vocab_path", default="prev_trained_model/electra_small/vocab.txt", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)

    parser.add_argument('--data_name', default='electra', type=str)
    parser.add_argument('--max_ngram', default=3, type=int)
    parser.add_argument("--do_data", default=True, action='store_true')
    parser.add_argument("--do_split", default=True, action='store_true')
    parser.add_argument("--do_lower_case", default=True, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--line_per_file", default=1000000000, type=int)
    parser.add_argument("--file_num", type=int, default=10,
                        help="Number of dynamic masking to pregenerate (with different masks)")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,  # 128 * 0.15
                        help="Maximum number of tokens to mask in each sequence")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.data_dir = Path(args.data_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir +"pregenerate_training_data_ngram.log")
    logger.info("pregenerate training data parameters:\n %s", args)
    tokenizer = BertTokenizer(vocab_file=args.vocab_path, do_lower_case=args.do_lower_case)

    # split big file
    if args.do_split:
        corpus_path =args.data_dir / "corpus/corpus.txt"
        split_save_path = args.data_dir / "corpus/train"
        if not split_save_path.exists():
            split_save_path.mkdir(exist_ok=True)
        line_per_file = args.line_per_file
        command = f'split -a 4 -l {line_per_file} -d {corpus_path} {split_save_path}/shard_'
        os.system(f"{command}")

    # generator train data
    if args.do_data:
        data_path = args.data_dir / "corpus/train"
        files = sorted([f for f in data_path.parent.iterdir() if f.exists() and '.txt' in str(f)])
        for idx in range(args.file_num):
            logger.info(f"pregenetate {args.data_name}_file_{idx}.json")
            save_filename = data_path / f"{args.data_name}_file_{idx}.json"
            num_instances = 0
            with save_filename.open('w',encoding="utf-8") as fw:
                for file_idx in range(len(files)):
                    file_path = files[file_idx]
                    file_examples = create_training_instances(input_file=file_path,
                                                              tokenizer=tokenizer,
                                                              max_seq_len=args.max_seq_len,
                                                              max_ngram=args.max_ngram,
                                                              short_seq_prob=args.short_seq_prob,
                                                              masked_lm_prob=args.masked_lm_prob,
                                                              max_predictions_per_seq=args.max_predictions_per_seq)
                    file_examples = [json.dumps(instance) for instance in file_examples]
                    for instance in file_examples:
                        fw.write(instance + '\n')
                        num_instances += 1
            metrics_file = data_path / f"{args.data_name}_file_{idx}_metrics.json"
            print(f"num_instances: {num_instances}")
            with metrics_file.open('w',encoding="utf-8") as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))

if __name__ == '__main__':
    main()

'''
python prepare_lm_data_ngram.py \
    --data_dir=dataset/ \
    --vocab_path=vocab.txt \
    --output_dir=outputs/ \
    --data_name=electra \
    --max_ngram=3 \
    --do_data
'''
