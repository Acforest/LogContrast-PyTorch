<div align="center">

# LogContrast: A Weakly Supervised Anomaly Detection Framework Leveraging Contrastive Learning

</div>

This repository provides the official code for the paper [LogContrast: A Weakly Supervised Anomaly Detection Framework Leveraging Contrastive Learning](https://ieeexplore.ieee.org/abstract/document/10191948/).

## Usage
### Requirements
This code is implemented with `Python 3.8.12`, `PyTorch 1.12.1` and `CUDA 11.6`.
- numpy==1.21.2
- pandas==1.5.1
- regex==2021.8.3
- scikit_learn==1.2.1
- torch==1.12.1
- tqdm==4.62.3
- transformers==4.15.0

### Installation
```shell
conda create -n LogContrast python=3.8
conda activate LogContrast
pip install -r requirments.txt
```

### Data Preperation
We need to download `HDFS` or `BGL` dataset.
```shell
# download `HDFS` dataset
sh ./datasets/HDFS/hdfs_download.sh
# download `BGL` dataset
sh ./datasets/BGL/bgl_download.sh
```

### Log Parsing
Then, parse raw logs for obtaining structured logs by following command:
```shell
# parsing `HDFS` dataset
sh ./scripts/hdfs_preprocess.sh
# parsing `BGL` dataset
sh ./scripts/bgl_preprocess.sh
```
The `datasets` folder will be structured as follows:
```shell
├── datasets/
│   ├── HDFS/     
|   |   ├── HDFS.log_structured.csv/
|   |   ├── HDFS.log_templates.csv/
|   |   ├── HDFS_logkey.json/
|   |   ├── HDFS_train_10000.csv/
|   |   ├── HDFS_test_575061.csv/
│   ├── BGL/
|   |   ├── BGL.log_structured.csv/
|   |   ├── BGL.log_templates.csv/
|   |   ├── BGL_logkey.json/
|   |   ├── BGL_train_10000.csv/
|   |   ├── BGL_test_942699.csv/
```

### Training & Testing
This is a training and testing demo for `LogContrast-sup0.2` on `HDFS` dataset.
If only for training, cancel the paramater `--do_test`, if only for testing, cancel the parameter `--do_train`.
```shell
python run.py \
--log_type "HDFS" \
--model_dir "./models/RQ2_1/HDFS_cl_bert_both_sup0.2_noise0.0/" \
--semantic_model_name "bert" \
--feat_type "both" \
--feat_dim 512 \
--vocab_size 120 \
--max_seq_len 128 \
--do_train \
--train_batch_size 64 \
--train_data_dir "./datasets/HDFS/HDFS_train_10000.csv" \
--loss_fct "cl" \
--num_epoch 20 \
--lr 0.00001 \
--weight_decay 0.01 \
--lambda_cl 0.1 \
--temperature 0.5 \
--sup_ratio 0.2 \
--noise_ratio 0.0 \
--do_test \
--test_batch_size 64 \
--test_data_dir "./datasets/HDFS/HDFS_test_575061.csv" \
--seed 1234 \
--device "cuda"
```
We provide the training and testing scripts in `scripts/` using a single GPU.

### Citation
```bib
@inproceedings{zhang2023logcontrast,
  title={LogContrast: A Weakly Supervised Anomaly Detection Method Leveraging Contrastive Learning},
  author={Zhang, Ting and Huang, Xin and Zhao, Wen and Mo, Guozhao and Bian, Shaohuang},
  booktitle={2023 IEEE 23rd International Conference on Software Quality, Reliability, and Security (QRS)},
  pages={48--59},
  year={2023},
  organization={IEEE}
}
```