import sys
import logging
import argparse
from prettytable import PrettyTable
import torch
from transformers import AutoModel, AutoTokenizer

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from model import CACSEModel


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def evaluate_ensemble(CACSE_model, UC_model):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default='bert-base',
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+',
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")

    args = parser.parse_args()

    CACSE_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer_UC = AutoTokenizer.from_pretrained('bert-base')
    UC_model = UC_model.to(device)

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                       'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                       'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = CACSE_tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )

            batch_UC = tokenizer_UC.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = CACSE_tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

            batch_UC = tokenizer_UC.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
            batch_UC[k] = batch_UC[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            outputs = CACSE_model(**batch, model_state="eval")
            outputs_UC = UC_model(**batch_UC)

        alpha = 1 / 4
        # alpha as a weighting factor to adjust the output of the two models
        return alpha * outputs.cpu() + (1 - alpha) * outputs_UC.last_hidden_state[:, 0].cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        AVG = sum([float(score) for score in scores]) / len(scores)

        return scores, AVG

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        AVG = sum([float(score) for score in scores]) / len(scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['Length', 'WordContent', 'Depth', 'TopConstituents',
                     'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                     'OddManOut', 'CoordinationInversion']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        return scores, AVG


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CACSE_bert = CACSEModel(Submodel_1_path='CACSE_BERT_submodel1',
                            Submodel_2_path='CACSE_BERT_submodel2',
                            num_head=6, hidden_dim=768, begin_encoder_layer=7,
                            constraint_trainstate_pos=4,
                            constraint_layer_pretrained='bert-base').to(device)
    # The parameters of CACSE initialised here must be the same as those saved earlier in train.py.

    CACSE_bert.load_state_dict(torch.load("output_CACSE/saved_ckpt/pytorch_model.bin"))
    # The checkpoint loaded here is saved after the training is done in train.py.

    UC_model = AutoModel.from_pretrained("ffgccInfoCSE-bert-base")
    # Download InfoCSE-base:https://huggingface.co/ffgcc/InfoCSE-bert-base/tree/main

    _, _ = evaluate_ensemble(CACSE_bert, UC_model)

# ensemble eval result:
# ------ test ------
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | 75.57 | 85.15 | 78.27 | 86.01 | 81.85 |    83.13     |      73.32      | 80.47 |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+