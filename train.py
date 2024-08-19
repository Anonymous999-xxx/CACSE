import argparse
from tqdm import tqdm
from loguru import logger
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import TrainDataset, TestDataset
from model import CACSEModel, InfoNCE_loss
from transformers import AutoTokenizer, AutoConfig
import os
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
import time
from evaluation import evaluation


def seed_everything(seed=42):
    '''
    Setting random seeds for the entire environment
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, dev_loader, optimizer, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = data['input_ids'].shape[-1]
            input_ids = data['input_ids'].view(-1, sql_len).to(device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)

            out_submodel1, out_submodel2, out_cascade = model(input_ids, attention_mask, token_type_ids, model_state="train")

            L1 = InfoNCE_loss(out_submodel1, device)
            L2 = InfoNCE_loss(out_submodel2, device)
            L3 = InfoNCE_loss(out_cascade, device) * args.hylambda

            loss = L1 + L2 + L3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.eval_step == 0:
                corrcoef = evaluate(model, dev_loader, device)
                logger.info('loss:{}, corrcoef: {} in step {} epoch {}'.format(loss, corrcoef, step, epoch))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('corrcoef', corrcoef, step)
                if best < corrcoef:
                    best = corrcoef
                    torch.save(model.state_dict(), join(args.output_path, 'pytorch_model.bin'))
                    # Only the model weights will be saved, and finally the two submodels will be saved as two separate models in huggingface format.
                    logger.info('higher corrcoef: {} in step {} epoch {}, save model'.format(best, step, epoch))
                model.train()


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
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids, model_state="eval")
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids, model_state="eval")
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def load_train_data_unsupervised(tokenizer, args):
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-unsupervise.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    with open(args.train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            feature = tokenizer([line, line], max_length=args.max_len, truncation=True, padding='max_length',
                                return_tensors='pt')
            # Two same sentences after dropout will have inconsistent encoding vectors
            feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_eval_data(tokenizer, args, mode):
    """
    Load STS-B validation set or test set
    """
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

    tokenizer = AutoTokenizer.from_pretrained(args.source_of_constraint_layer)

    model = CACSEModel(Submodel_1_path=args.submodel1_path, Submodel_2_path=args.submodel2_path,
                       num_head=args.cross_head_num, hidden_dim=768, begin_encoder_layer=args.begin_layer,
                       constraint_trainstate_pos=args.constraint_trainstate_pos,
                       constraint_layer_pretrained=args.source_of_constraint_layer).to(args.device)

    if args.do_train:
        # 加载数据集
        train_data = load_train_data_unsupervised(tokenizer, args)
        train_dataset = TrainDataset(train_data, tokenizer, max_len=args.max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_data = load_eval_data(tokenizer, args, 'dev')
        dev_dataset = TestDataset(dev_data, tokenizer, max_len=args.max_len)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                    num_workers=args.num_workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train(model, train_dataloader, dev_dataloader, optimizer, args)


    if args.do_predict:
        test_data = load_eval_data(tokenizer, args, 'test')
        test_dataset = TestDataset(test_data, tokenizer, max_len=args.max_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                     num_workers=args.num_workers)
        model.load_state_dict(torch.load(join(args.output_path, 'pytorch_model.bin')))
        model.eval()
        corrcoef = evaluate(model, test_dataloader, args.device)
        logger.info('testset corrcoef:{}'.format(corrcoef))
        _, _ = evaluation(model)

        #     savetohuggingface
        submodel1 = model.Submodel_1
        submodel2 = model.Submodel_2
        config_submodel1 = AutoConfig.from_pretrained(args.submodel1_path)
        config_submodel2 = AutoConfig.from_pretrained(args.submodel2_path)

        submodel1.save_pretrained(os.path.join(args.output_path, "submodel1"))
        config_submodel1.save_pretrained(os.path.join(args.output_path, "submodel1"))

        submodel2.save_pretrained(os.path.join(args.output_path, "submodel2"))
        config_submodel2.save_pretrained(os.path.join(args.output_path, "submodel2"))
        logger.info("model saved done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--submodel1_path", type=str,
                        default='CACSE_BERT_submodel1')
    parser.add_argument("--submodel2_path", type=str,
                        default='CACSE_BERT_submodel2')
    parser.add_argument("--source_of_constraint_layer", type=str,
                        default='bert-base')
    parser.add_argument("--cross_head_num", type=int, default=6)
    parser.add_argument("--begin_layer", type=int, default=7,
                        help="Layers of the encoder corresponding to the first layer of the cross-attention cascade for the submodel")
    parser.add_argument("--constraint_trainstate_pos", type=int, default=4, help='')
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument("--output_path", type=str, default='output_CACSE')
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=64, help="max length of input")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--hylambda", type=float, default=1.0)
    parser.add_argument("--train_file", type=str, default="data/Wiki_for_CACSE.txt")
    parser.add_argument("--dev_file", type=str, default="data/stsbenchmark/sts-dev.csv")
    parser.add_argument("--test_file", type=str, default="data/stsbenchmark/sts-test.csv")

    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=False) # if your want to train CACSE-BERT, please change to 1 or True.
    parser.add_argument("--do_predict", action='store_true', default=True)

    args = parser.parse_args()
    seed_everything(seed=args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    args.output_path = join(args.output_path, "saved_ckpt")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    main(args)
