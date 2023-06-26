import os
import sys

import math
import random
import time
import warnings

import numpy as np
import pandas as pd
import re

import torch
import torch.nn as nn
import transformers as T
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import TruncationStrategy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.notebook import tqdm

def seed_torch(seed=42):
    # python の組み込み関数の seed を固定
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy の seed を固定
    np.random.seed(seed)
    # torch の seed を固定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定論的アルゴリズムを使用する
    torch.backends.cudnn.deterministic = True

class CreateDataset(Dataset):
    def __init__(self, X, model_name, include_labels=True):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.X = X
        self.include_labels = include_labels

        sentences = X["title_abstract"].tolist()

        max_length = 256
        self.encoded = tokenizer.batch_encode_plus(
            sentences,
            padding = 'max_length',            
            max_length = max_length,
            truncation = True,
            return_attention_mask=True
        )
        if self.include_labels:
            self.labels = X["judgement"].values 

    def __len__(self):  
        return len(self.X)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encoded['input_ids'][idx])
        attention_mask = torch.tensor(self.encoded['attention_mask'][idx])

        if self.include_labels:
            label = torch.tensor(self.labels[idx]).float()
            return input_ids, attention_mask, label

        return input_ids, attention_mask

def get_train_data(train, Fold, seed):

    # 交差検証 用の番号を振ります。
    fold = StratifiedKFold(n_splits=Fold, shuffle=True, random_state=seed)
    for n, (train_index, val_index) in enumerate(fold.split(train, train["judgement"])):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(np.uint8)

    return train

def get_test_data(test):
    return test

class BertModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.sigmoid(out.logits).squeeze()

        return out

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))

def train_fn(train_loader, model, criterion, optimizer, epoch, device):
    start = end = time.time()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        y_preds = model(input_ids, attention_mask)

        loss = criterion(y_preds, labels)

        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()

        optimizer.step()

        if step % 100 == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    start = end = time.time()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()
    preds = []

    for step, (input_ids, attention_mask, labels) in enumerate(valid_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(input_ids, attention_mask)

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # record score
        preds.append(y_preds.to("cpu").numpy())

        if step % 100 == 0 or step == (len(valid_loader) - 1):
            print(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    predictions = np.concatenate(preds)
    return losses.avg, predictions

def inference(Fold, LOGGER, model_name, device, eval):
    predictions = []

    eval_dataset = CreateDataset(eval, model_name, include_labels=False)
    eval_loader = DataLoader(
        eval_dataset, batch_size=16, shuffle=False, pin_memory=True
    )

    for fold in range(Fold):
        LOGGER.info(f"========== model: bert-base-uncased fold: {fold} inference ==========")
        model = BertModel(model_name)
        model.to(device)
        model.load_state_dict(torch.load(f"bert-base-uncased_fold{fold}_best.pth")["model"])
        model.eval()
        preds = []
        for i, (input_ids, attention_mask) in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                y_preds = model(input_ids, attention_mask)
            preds.append(y_preds.to("cpu").numpy())
        preds = np.concatenate(preds)
        predictions.append(preds)
    predictions = np.mean(predictions, axis=0)

    return predictions

def init_logger(log_file="train.log"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def train_loop(train, fold, LOGGER, model_name, device, border):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # Data Loader
    # ====================================================
    trn_idx = train[train["fold"] != fold].index
    val_idx = train[train["fold"] == fold].index

    train_folds = train.loc[trn_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)

    train_dataset = CreateDataset(train_folds, model_name)
    valid_dataset = CreateDataset(valid_folds, model_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # Model
    # ====================================================
    model = BertModel(model_name)
    model.to(device)

    optimizer = T.AdamW(model.parameters(), lr=2e-5)

    criterion = nn.BCELoss()

    # ====================================================
    # Loop
    # ====================================================
    best_score = -1
    best_loss = np.inf

    for epoch in range(5):
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds["judgement"].values

        # scoring
        score = fbeta_score(valid_labels, np.where(preds < border, 0, 1), beta=7.0)

        elapsed = time.time() - start_time
        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score}")

        if score > best_score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds}, f"bert-base-uncased_fold{fold}_best.pth"
            )

    check_point = torch.load(f"bert-base-uncased_fold{fold}_best.pth")

    valid_folds["preds"] = check_point["preds"]

    return valid_folds

def get_result(result_df, border, LOGGER):
    preds = result_df["preds"].values
    labels = result_df["judgement"].values
    score = fbeta_score(labels, np.where(preds < border, 0, 1), beta=7.0)
    LOGGER.info(f"Score: {score:<.5f}")


def main():
    train = pd.read_csv("train.csv")
    eval = pd.read_csv("test.csv")
    sub = pd.read_csv("sample_submit.csv", header=None)
    sub.columns = ["id", "judgement"]

    train["judgement"][2488] = 0
    train["judgement"][7708] = 0

    seed = 471
    seed_torch(seed)

    model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'

    train["title_abstract"] = train["title"] + " " + train["abstract"].fillna("")

    Fold = 10

    train = get_train_data(train, Fold, seed)

    eval["title_abstract"] = eval["title"] + " " + eval["abstract"].fillna("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    border = len(train[train["judgement"] == 1]) / len(train["judgement"])
    print(border)

    LOGGER = init_logger()

    # Training
    oof_df = pd.DataFrame()
    for fold in range(Fold):
        _oof_df = train_loop(train, fold, LOGGER, model_name, device, border)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df, border, LOGGER)

    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df, border, LOGGER)

    # Save OOF result
    oof_df.to_csv("oof_df.csv", index=False)

    # Inference
    predictions = inference(Fold, LOGGER, model_name, device, eval)
    pd.Series(predictions).to_csv("predictions3.csv", index=False)
    predictions1 = np.where(predictions < 0.0262, 0, 1)

    # submission
    sub["judgement"] = predictions1
    sub.to_csv("submission3.csv", index=False, header=False)

if __name__ == "__main__":
    main()
