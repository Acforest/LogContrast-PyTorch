import os
import sys
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.nn import functional as F
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score
from config import load_configs
from dataset import load_data
from loss import CELoss, CLLoss
from model import LogContrast
from utils import set_seed


def train(model, dataloader, criterion, optimizer, best_metric_to_save, device):
    all_losses = defaultdict(list)
    all_logits, all_probs, all_labels = [], [], []
    model.train()
    for batch in tqdm(dataloader, desc='Training'):
        semantics = {k: v.to(device) for k, v in batch['semantics'].items()}
        sequences = batch['sequences'].to(device)
        seqence_masks = batch['sequence_masks'].to(device)
        true_labels = batch['true_labels'].to(device)
        train_labels = batch['train_labels'].to(device)

        logits, feats, feats_aug = model(semantics, sequences, seqence_masks)

        if isinstance(criterion, CELoss):
            losses = criterion(logits, train_labels)
        elif isinstance(criterion, CLLoss):
            losses = criterion(logits, train_labels, feats, feats_aug)
        else:
            raise ValueError('`criterion` must be ["ce_loss", "cl_loss"]')

        for k, v in losses.items():
            all_losses[k].append(v.item())

        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

        probs = F.softmax(logits, dim=1)
        all_logits += logits.detach().cpu().numpy().tolist()
        all_probs += probs[:, 1].detach().cpu().numpy().tolist()
        all_labels += true_labels.detach().cpu().numpy().tolist()

    result_dict = defaultdict()
    result_dict['losses'] = defaultdict(float)
    result_dict['metrics'] = defaultdict(float)
    for k, v in all_losses.items():
        result_dict['losses'][k] = np.nanmean(v)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    result_dict['metrics']['auc'] = roc_auc_score(all_labels, all_probs)
    for th in thresholds:
        all_preds = [1 if prob >= th else 0 for prob in all_probs]
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        if best_metric_to_save == 'precision':
            if precision > result_dict['metrics'][best_metric_to_save]:
                result_dict['threshold'] = th
                result_dict['metrics']['precision'] = precision
                result_dict['metrics']['recall'] = recall
                result_dict['metrics']['f1'] = f1
                result_dict['metrics']['accuracy'] = accuracy
        elif best_metric_to_save == 'recall':
            if recall > result_dict['metrics'][best_metric_to_save]:
                result_dict['threshold'] = th
                result_dict['metrics']['precision'] = precision
                result_dict['metrics']['recall'] = recall
                result_dict['metrics']['f1'] = f1
                result_dict['metrics']['accuracy'] = accuracy
        elif best_metric_to_save == 'f1':
            if f1 > result_dict['metrics'][best_metric_to_save]:
                result_dict['threshold'] = th
                result_dict['metrics']['precision'] = precision
                result_dict['metrics']['recall'] = recall
                result_dict['metrics']['f1'] = f1
                result_dict['metrics']['accuracy'] = accuracy
        elif best_metric_to_save == 'accuracy':
            if accuracy > result_dict['metrics'][best_metric_to_save]:
                result_dict['threshold'] = th
                result_dict['metrics']['precision'] = precision
                result_dict['metrics']['recall'] = recall
                result_dict['metrics']['f1'] = f1
                result_dict['metrics']['accuracy'] = accuracy
        else:
            raise ValueError('`best_metric_to_save` must be ["precision", "recall", "f1", "accuracy"]')
    return result_dict


def test(model, dataloader, criterion, threshold, device):
    all_losses = defaultdict(list)
    all_logits, all_probs, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            semantics = {k: v.to(device) for k, v in batch['semantics'].items()}
            sequences = batch['sequences'].to(device)
            seqence_masks = batch['sequence_masks'].to(device)
            true_labels = batch['true_labels'].to(device)
            train_labels = batch['train_labels'].to(device)

            logits, feats, feats_aug = model(semantics, sequences, seqence_masks)

            if isinstance(criterion, CELoss):
                losses = criterion(logits, train_labels)
            elif isinstance(criterion, CLLoss):
                losses = criterion(logits, train_labels, feats, feats_aug)
            else:
                raise ValueError('`criterion` must be ["ce_loss", "cl_loss"]')

            for k, v in losses.items():
                all_losses[k].append(v.item())

            probs = F.softmax(logits, dim=1)
            all_logits += logits.detach().cpu().numpy().tolist()
            all_probs += probs[:, 1].detach().cpu().numpy().tolist()
            all_labels += true_labels.detach().cpu().numpy().tolist()

    all_preds = [1 if prob >= threshold else 0 for prob in all_probs]
    result_dict = defaultdict()
    result_dict['losses'] = defaultdict(float)
    result_dict['metrics'] = defaultdict(float)
    for k, v in all_losses.items():
        result_dict['losses'][k] = np.nanmean(v)
    result_dict['probs'] = all_probs
    result_dict['preds'] = all_preds
    result_dict['labels'] = all_labels
    result_dict['threshold'] = threshold
    result_dict['fpr'], result_dict['tpr'], result_dict['thresholds'] = roc_curve(all_labels, all_probs, pos_label=1)
    result_dict['metrics']['auc'] = roc_auc_score(all_labels, all_probs)
    result_dict['metrics']['precision'] = precision_score(all_labels, all_preds)
    result_dict['metrics']['recall'] = recall_score(all_labels, all_preds)
    result_dict['metrics']['f1'] = f1_score(all_labels, all_preds)
    result_dict['metrics']['accuracy'] = accuracy_score(all_labels, all_preds)
    return result_dict


if __name__ == '__main__':
    args = load_configs()

    assert args.do_train or args.do_test, '`do_train` and `do_test` should be at least true for one'

    os.makedirs(args.model_dir, exist_ok=True)

    log_name = f'{args.log_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    args.log_dir = os.path.join(args.model_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(args.log_dir))

    args_dict = vars(args)
    logger.info(f'Parameters: {args_dict}')
    with open(os.path.join(args.model_dir, 'configs.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    args.device = torch.device(args.device)

    set_seed(args.seed)

    if args.semantic_model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.semantic_model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
    elif args.semantic_model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    else:
        raise ValueError('`semantic_model_name` must be in ["bert", "roberta", "albert"]')

    train_dataloader = load_data(tokenizer=tokenizer,
                                 data_dir=args.train_data_dir,
                                 batch_size=args.train_batch_size,
                                 max_seq_len=args.max_seq_len,
                                 sup_ratio=args.sup_ratio,
                                 noise_ratio=args.noise_ratio,
                                 evo_ratio=0.0)
    test_dataloader = load_data(tokenizer=tokenizer,
                                data_dir=args.test_data_dir,
                                batch_size=args.test_batch_size,
                                max_seq_len=args.max_seq_len,
                                sup_ratio=1.0,
                                noise_ratio=0.0,
                                evo_ratio=args.evo_ratio)

    model = LogContrast(vocab_size=args.vocab_size,
                        feat_dim=args.feat_dim,
                        feat_type=args.feat_type,
                        semantic_model_name=args.semantic_model_name,
                        max_seq_len=args.max_seq_len,
                        dropout_p=args.dropout_p)
    model.to(args.device)

    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.loss_fct == 'ce':
        criterion = CELoss()
    elif args.loss_fct == 'cl':
        criterion = CLLoss(temperature=args.temperature, lambda_cl=args.lambda_cl)
    else:
        raise ValueError('`loss_fct` must be in ["ce", "cl"]')

    if args.do_train:
        if args.load_model:
            ckpt = torch.load(os.path.join(args.model_dir, 'model.pth'))
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
        best_results = defaultdict()
        best_results['losses'] = defaultdict(float)
        best_results['metrics'] = defaultdict(float)
        logger.info('Training start:')
        for epoch in tqdm(range(args.num_epochs), desc='Epoch'):
            train_results = train(model, train_dataloader, criterion, optimizer, args.best_metric_to_save, args.device)
            logger.info('[train]')
            logger.info(f'epoch: {epoch + 1}/{args.num_epochs} - {100 * (epoch + 1) / args.num_epochs:.2f}%')
            logger.info('[losses]')
            for k, v in train_results['losses'].items():
                logger.info(f'{k}: {v}')
            logger.info('[metrics]')
            logger.info(f'threshold: {train_results["threshold"]}')
            for k, v in train_results['metrics'].items():
                logger.info(f'{k}: {v}')
            if best_results['metrics'][args.best_metric_to_save] < train_results['metrics'][args.best_metric_to_save]:
                best_results.update(train_results)
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'threshold': best_results['threshold']
                }
                torch.save(ckpt, os.path.join(args.model_dir, 'model.pth'))
                logger.info(f'model saved at "{os.path.join(args.model_dir, "model.pth")}"')

    if args.do_test:
        ckpt = torch.load(os.path.join(args.model_dir, 'model.pth'))
        model.load_state_dict(ckpt['model'])
        threshold = ckpt['threshold']
        logger.info('Testing start:')
        test_results = test(model, test_dataloader, criterion, threshold, args.device)
        logger.info('[test]')
        logger.info('[metrics]')
        logger.info(f'threshold: {threshold}')
        for k, v in test_results['metrics'].items():
            logger.info(f'{k}: {v}')
        with open(os.path.join(args.model_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)

    logger.info(f'log saved at: "{args.log_dir}"')
