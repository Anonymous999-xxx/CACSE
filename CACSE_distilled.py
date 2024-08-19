import argparse
from tqdm import tqdm
from loguru import logger
import numpy as np
from scipy.stats import spearmanr
from shutil import copyfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from dataset import TrainDataset, TestDataset
from model import CACSEModel, CACSE_distilled
import os
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
import time
from transformers import AutoModel, AutoTokenizer
from ensemble_CACSE_UC import evaluate_ensemble
from evaluation import eval_distilledmodel,eval_cacse_self_distilledmodel,evaluation

torch.set_printoptions(profile="default")  # reset


def seed_everything(seed=42):
    '''
    Setting the seed for the entire development environment:
    param seed:
    param device:
    return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def distill_loss(y_pred, out0, out_t, device, temp=0.05, k1=0.01):
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temp

    loss1 = F.cross_entropy(sim, y_true)
    k2 = 1 - k1
    L1_loss = torch.nn.L1Loss()

    loss2 = L1_loss(out0, out_t)
    loss_all = k1 * loss1 + loss2 * k2

    return loss_all


def train(model, teacher_casce, train_loader, dev_loader, optimizer, args):
    logger.info("start training")

    model.train()

    device = args.device
    best = 0
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = data['input_ids'].shape[-1]
            input_ids = data['input_ids'].view(-1, sql_len).to(device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)
            model = model.to(device)

            out_distilled = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            with torch.no_grad():
                out_cacse = teacher_casce(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids, model_state='eval')

            out_all = out_cacse
            out = torch.cat((out_all, out_distilled), dim=0)  # true,pred,...

            for i in range(len(out)):
                if i % 2 == 0:
                    out[i] = out_all[i // 2]
                    out[i + 1] = out_distilled[i // 2]

            loss = distill_loss(out, out_distilled, out_all, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.eval_step == 0:
                corrcoef = evaluate(model, dev_loader, device)
                logger.info('loss:{}, corrcoef: {} in step {} epoch {}'.format(loss, corrcoef, step, epoch))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('corrcoef', corrcoef, step)
                model.train()
                if best < corrcoef:
                    best = corrcoef
                    torch.save(model.state_dict(), join(args.output_path, 'pytorch_model.bin'))
                    logger.info('higher corrcoef: {} in step {} epoch {}, save model'.format(best, step, epoch))

        if args.do_predict:
            test_data = load_eval_data(tokenizer, args, 'test')
            test_dataset = TestDataset(test_data, tokenizer, max_len=args.max_len)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                         num_workers=args.num_workers)
            model.load_state_dict(torch.load(join(args.output_path, 'pytorch_model.bin')))
            model.eval()
            corrcoef = evaluate(model, test_dataloader, args.device)
            logger.info('testset corrcoef:{}'.format(corrcoef))


def evaluate(model, dataloader, device):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            source_pred = model(input_ids=source_input_ids, attention_mask=source_attention_mask,
                                token_type_ids=source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            target_pred = model(input_ids=target_input_ids, attention_mask=target_attention_mask,
                                token_type_ids=target_token_type_ids)

            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))

    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def load_train_data_unsupervised(tokenizer, args):
    inputlist = []
    feature_list = []
    src = []
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-unsupervise.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            f.close()
            return feature_list

    with open(args.train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            src.append(line)
        f.close()
    for i in range(len(src)):
        inputlist.append([src[i]])
    for k in range(len(inputlist)):
        feature = tokenizer(inputlist[k], max_length=args.max_len, truncation=True, padding='max_length',
                            return_tensors='pt')
        feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
        f.close()
    return feature_list


def load_eval_data(tokenizer, args, mode):
    assert mode in ['dev', 'test'], 'mode should in ["dev", "test"]'
    logger.info('loading {} data'.format(mode))
    output_path = os.path.dirname(args.output_path)
    eval_file_cache = join(output_path, '{}.pkl'.format(mode))
    if os.path.exists(eval_file_cache) and not args.overwrite_cache:
        with open(eval_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of {} data:{}".format(mode, len(feature_list)))
            return feature_list

    if mode == 'dev':
        eval_file = args.dev_file
    else:
        eval_file = args.test_file
    feature_list = []
    with open(eval_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        logger.info("len of {} data:{}".format(mode, len(lines)))
        for line in tqdm(lines):
            line = line.strip().split("\t")
            assert len(line) == 7 or len(line) == 9
            score = float(line[4])
            data1 = tokenizer(line[5].strip(), max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')
            data2 = tokenizer(line[6].strip(), max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')

            feature_list.append((data1, data2, score))
    with open(eval_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    model = CACSE_distilled(args.pretrain_model_path,pooling='cls',dropout=args.dropout).to(args.device)

    teacher_CACSE = CACSEModel(Submodel_1_path="CACSE_BERT_submodel1",
                               Submodel_2_path='CACSE_BERT_submodel2',
                               num_head=6, hidden_dim=768, begin_encoder_layer=7,
                               constraint_trainstate_pos=4,
                               constraint_layer_pretrained="bert-base").to(
        args.device)
    teacher_CACSE.load_state_dict(torch.load('output_CACSE/saved_ckpt/pytorch_model.bin'))

    for param in teacher_CACSE.parameters():
        param.requires_grad = False

    evaluation(teacher_CACSE)
    # teacher result:
    # ------ test ------
    # +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    # | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    # +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    # | 75.38 | 83.23 | 76.45 | 84.14 | 79.66 |    81.29     |      72.98      | 79.02 |
    # +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    if args.do_train:
        train_data = load_train_data_unsupervised(tokenizer, args)
        train_dataset = TrainDataset(train_data, tokenizer, max_len=args.max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_data = load_eval_data(tokenizer, args, 'dev')
        dev_dataset = TestDataset(dev_data, tokenizer, max_len=args.max_len)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                    num_workers=args.num_workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train(model, teacher_CACSE, train_dataloader, dev_dataloader, optimizer, args)
    if args.do_predict:
        model.load_state_dict(torch.load(join(args.output_path, 'pytorch_model.bin')))
        model.eval()
        eval_cacse_self_distilledmodel(model)
        # disilled model eval result:
        # ------ test ------
        # +-------+-------+-------+-------+-------+--------------+-----------------+-------+
        # | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
        # +-------+-------+-------+-------+-------+--------------+-----------------+-------+
        # | 74.75 | 83.74 | 76.23 | 83.87 | 79.58 |    81.18     |      73.58      | 78.99 |
        # +-------+-------+-------+-------+-------+--------------+-----------------+-------+

        model_distilled_saved=model.CACSE_distilled_model
        config_saved=model.config
        tokenizer_save=tokenizer

        model_distilled_saved.save_pretrained(join(args.output_path,'cacse_distilled'))
        config_saved.save_pretrained(join(args.output_path,'cacse_distilled'))
        tokenizer_save.save_pretrained(join(args.output_path,'cacse_distilled'))
        logger.info("model saved done!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="g pu or cpu")
    parser.add_argument("--output_path", type=str, default='output_CACSE_self_distilled', help="output path")
    parser.add_argument("--lr", type=float, default=10e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=64, help="max length of input")
    parser.add_argument("--seed", type=int, default=42, help="random seed")  # 42
    parser.add_argument("--train_file", type=str, default="data/Wiki_for_CACSE.txt")
    parser.add_argument("--dev_file", type=str, default="data/stsbenchmark/sts-dev.csv")
    parser.add_argument("--test_file", type=str, default="data/stsbenchmark/sts-test.csv")
    parser.add_argument("--pretrain_model_path", type=str,default="bert-base")
    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=0)# if your want to train CACSE-BERT-distilled, please change to 1 or True.
    parser.add_argument("--do_predict", action='store_true', default=1)

    args = parser.parse_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    args.output_path = join(args.output_path, "saved_ckpt")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    main(args)