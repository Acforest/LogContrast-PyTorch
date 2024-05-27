import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import set_seed
from collections import defaultdict
from logparser import Spell, Drain


# the hyperparameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf

class LogPreProcessor:
    """Class for log pre-processing (log parsing)"""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 log_type: str,
                 log_file: str,
                 parser_type: str,
                 random_seed: int,
                 train_num: int) -> None:
        self.log_type = log_type
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_file = log_file
        self.parser_type = parser_type
        self.random_seed = random_seed
        self.train_num = train_num

        if self.log_type == 'HDFS':
            log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
            regex = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
            if self.parser_type == 'Spell':
                tau = 0.7
                self.parser = Spell.LogParser(indir=self.input_dir, outdir=self.output_dir,
                                              log_format=log_format, tau=tau, rex=regex,
                                              keep_para=False)
            elif parser_type == 'Drain':
                st = 0.5
                depth = 4
                self.parser = Drain.LogParser(indir=self.input_dir, outdir=self.output_dir,
                                              log_format=log_format, depth=depth,
                                              st=st, rex=regex, keep_para=False)
        elif self.log_type == 'BGL':
            log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
            regex = [r'core\.\d+']
            if parser_type == 'Spell':
                tau = 0.75
                self.parser = Spell.LogParser(indir=self.input_dir, outdir=self.output_dir,
                                              log_format=log_format, tau=tau, rex=regex,
                                              keep_para=False)
            elif parser_type == 'Drain':
                st = 0.5
                depth = 4
                self.parser = Drain.LogParser(indir=self.input_dir, outdir=self.output_dir,
                                              log_format=log_format, depth=depth,
                                              st=st, rex=regex, keep_para=False)
        elif self.log_type == 'Thunderbird':
            log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
            regex = [r'(\d+\.){3}\d+']
            if parser_type == 'Spell':
                tau = 0.5
                self.parser = Spell.LogParser(indir=self.input_dir, outdir=self.output_dir,
                                              log_format=log_format, tau=tau, rex=regex,
                                              keep_para=False)
            elif parser_type == 'Drain':
                st = 0.5
                depth = 4
                self.parser = Drain.LogParser(indir=self.input_dir, outdir=self.output_dir,
                                              log_format=log_format, depth=depth, st=st,
                                              rex=regex, keep_para=False)
        else:
            raise ValueError('log_type is not in ["HDFS", "BGL", "Thunderbird"]')

    def parse_log(self) -> None:
        self.parser.parse(self.log_file)

    def generate_structured_dataset(self) -> None:
        log_structured_file = os.path.join(self.output_dir, f'{self.log_file}_structured.csv')
        data_df = pd.read_csv(log_structured_file, engine='c', na_filter=False, memory_map=True)

        if self.log_type == 'HDFS':
            # session windows
            label_df = pd.read_csv(os.path.join(self.input_dir, 'anomaly_label.csv'))
            label_dict = defaultdict(str)
            event_id_seq_dict = defaultdict(list)
            event_template_seq_dict = defaultdict(set)
            event_id_map = defaultdict(int)
            cur_id = 1
            for idx, row in tqdm(label_df.iterrows(), desc='Reading `anomaly_label.csv`'):
                label_dict[row['BlockId']] = row['Label']
            for idx, row in tqdm(data_df.iterrows(), desc=f'Reading `{self.log_file}_structured.csv`'):
                blk_id_list = re.findall(r'(blk_-?\d+)', row['Content'])
                blk_id_set = set(blk_id_list)
                for blk_id in blk_id_set:
                    if row['EventId'] not in event_id_map.keys():
                        event_id_map[row['EventId']] = cur_id
                        cur_id += 1
                    event_id_seq_dict[blk_id].append(event_id_map[row['EventId']])
                    event_template_seq_dict[blk_id].add(row['EventTemplate'])
            event_id_seq_df = pd.DataFrame(list(event_id_seq_dict.items()), columns=['BlockId', 'EventIdSequence'])
            event_template_seq_df = pd.DataFrame(list(event_template_seq_dict.items()),
                                                 columns=['BlockId', 'EventTemplateSequence'])
            data_df = pd.concat([event_id_seq_df, event_template_seq_df['EventTemplateSequence']], axis=1)

            data_df['EventTemplateSequence'] = data_df['EventTemplateSequence']
            data_df['Label'] = data_df['BlockId'].apply(lambda r: label_dict[r])

        elif self.log_type == 'BGL' or self.log_type == 'Thunderbird':
            data_df['Datetime'] = pd.to_datetime(data_df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
            data_df['Timestamp'] = data_df["Datetime"].values.astype(np.int64) // 10 ** 9
            data_df['DeltaTime'] = (data_df['Datetime'].diff() / np.timedelta64(1, 's')).fillna(0)
            data_df["Label"] = data_df["Label"].apply(lambda x: 'Normal' if x == '-' else 'Anomaly')

            event_id_map = defaultdict(int)
            cur_id = 1
            fixed_window_size = 5
            line_id_list, event_id_list, event_template_set, delta_time_list, label_list = [], [], set(), [], []
            row_list = []
            for idx, row in tqdm(data_df.iterrows(), desc=f'Reading `{self.log_file}_structured.csv`'):
                if row['EventId'] not in event_id_map.keys():
                    event_id_map[row['EventId']] = cur_id
                    cur_id += 1
                line_id_list.append(row['LineId'])
                event_id_list.append(event_id_map[row['EventId']])
                event_template_set.add(row['EventTemplate'])
                delta_time_list.append(row['DeltaTime'])
                label_list.append(row['Label'])
                if (idx + 1) % fixed_window_size == 0:
                    row_list.append({
                        'LineId': line_id_list,
                        'EventIdSequence': event_id_list,
                        'EventTemplateSequence': event_template_set,
                        'DeltaTimeSequence': delta_time_list,
                        'LabelSequence': label_list,
                        'Label': 'Anomaly' if 'Anomaly' in label_list else 'Normal'
                    })
                    line_id_list, event_id_list, event_template_set, delta_time_list, label_list = [], [], set(), [], []
            if line_id_list:
                row_list.append({
                    'LineId': line_id_list,
                    'EventIdSequence': event_id_list,
                    'EventTemplateSequence': list(event_template_set),
                    'DeltaTimeSequence': delta_time_list,
                    'LabelSequence': label_list,
                    'Label': 'Anomaly' if 'Anomaly' in label_list else 'Normal'
                })
            data_df = pd.DataFrame(row_list)
        else:
            raise ValueError('`log_type` must be in ["HDFS", "BGL", "Thunderbird"]')

        data_df['EventTemplateSequence'] = data_df['EventTemplateSequence'].apply(lambda x: ' '.join(''.join(x)
                                                                                  .replace('<*>', '').replace('*', '')
                                                                                  .replace('/', '').replace(':', '')
                                                                                  .replace('.', ' ').split()))

        if self.train_num > 0:
            train_data_df = data_df.sample(n=self.train_num, random_state=self.random_seed)
            train_data_df.to_csv(os.path.join(self.output_dir, f'{self.log_type}_train_{len(train_data_df)}.csv'), index=False)
        data_df.to_csv(os.path.join(self.output_dir, f'{self.log_type}_test_{len(data_df)}.csv'), index=False)
        with open(os.path.join(self.output_dir, f'{self.log_type}_logkey.json'), "w", encoding='utf-8') as f:
            json.dump(event_id_map, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Command line interface for log parser')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='The diretory of input log')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The diretory of output file')
    parser.add_argument('--log_type', type=str, required=True, choices=['HDFS', 'BGL', 'Thunderbird'],
                        help='The type of input log file')
    parser.add_argument('--log_file', type=str, required=True,
                        help='The name of input log file')
    parser.add_argument('--parser_type', type=str, default='Drain', choices=['Spell', 'Drain'],
                        help='The type of log parser')
    parser.add_argument('--random_seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--train_num', type=int, default=10000,
                        help='Number of data for training')

    args = parser.parse_args()

    set_seed(args.random_seed)

    log_preprocessor = LogPreProcessor(input_dir=args.input_dir,
                                       output_dir=args.output_dir,
                                       log_type=args.log_type,
                                       log_file=args.log_file,
                                       parser_type=args.parser_type,
                                       random_seed=args.random_seed,
                                       train_num=args.train_num)

    log_preprocessor.parse_log()
    log_preprocessor.generate_structured_dataset()


if __name__ == '__main__':
    main()
