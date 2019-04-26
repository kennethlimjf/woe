# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import pandas as pd
import yaml

class config:

    def __init__(self):
        self.config = None
        self.dataset_train = None
        self.variable_type = None
        self.bin_var_list = None
        self.discrete_var_list = None
        self.candidate_var_list = None
        self.dataset_len = None
        self.min_sample = None
        self.global_bt = None
        self.global_gt = None

    def load_file(self, config_path):
        self.config = pd.read_csv(config_path)

        # specify variable dtypes
        self.variable_type = self.config[['var_name', 'var_dtype']]
        self.variable_type = self.variable_type.rename(columns={'var_name': 'v_name', 'var_dtype': 'v_type'})
        self.variable_type = self.variable_type.set_index(['v_name'])

        # specify the list of continuous variable to be splitted into bin
        self.bin_var_list = self.config[self.config['is_tobe_bin'] == 1]['var_name']
        # specify the list of discrete variable to be merged into supper classes
        self.discrete_var_list = self.config[(self.config['is_candidate'] == 1) & (self.config['var_dtype'] == 'object')]['var_name']

        # specify the list of model input variable
        self.candidate_var_list = self.config[self.config['is_candidate'] == 1]['var_name']

    def load_min_sample_weight_config(self, config_path):
        with open(config_path, 'r') as f:
            self.min_sample_weight_config = yaml.load(f)['features']

    def get_min_sample(self, var):
        min_sample_weight = None
        for feature in self.min_sample_weight_config:
            if feature.get('feature_name') == var:
                min_sample_weight = feature.get('min_sample_weight')

        if min_sample_weight:
            return int(self.dataset_len * min_sample_weight)
        else:
            raise ValueError('Feature not found')

    def set_dataset(self, df):
        self.dataset_train = df.copy()
        self.dataset_train.columns = [col.split('.')[-1] for col in self.dataset_train.columns]

        # specify some other global variables about the training dataset
        self.dataset_len = len(self.dataset_train)
        # self.min_sample = int(self.dataset_len * self.min_sample_weight)

        if (('target' in self.dataset_train.columns) and
            not self.dataset_train.target.isnull().any()):
            self.global_bt = sum(self.dataset_train['target'])
            self.global_gt = len(self.dataset_train) - sum(self.dataset_train['target'])

    def change_config_var_dtype(self,var_name,type,inplace_file=True):
        if type in ['object','string','int64','uint8','float64','bool1','bool2','dates','category']:
            self.variable_type.loc[var_name,'v_type'] = type
        else:
            raise KeyError("Invalid dtype specified! ")
