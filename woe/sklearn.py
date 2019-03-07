import contextlib
import os
import pandas as pd
import subprocess
import woe.feature_process as fp


FIT_DATA_PATH = 'woe_processing/fit_data.csv'
TRANSFORM_DATA_PATH = 'woe_processing/transform_data.csv'
TRANSFORMED_WOE_DATA_PATH = 'woe_processing/transform_data_woe.csv'
FEATURE_DETAILS_OUTPUT = 'woe_processing/woe_feature_details.csv'


class WoeTransformer():
    """
    WOE Transformer based on scikit-learn API
    """

    def __init__(self,
                 config_filepath=None,
                 save_woe_pickle_filepath=None,
                 load_woe_pickle_filepath=None,
                 min_sample_weight=0.05):
        self.config_filepath = config_filepath
        self.save_woe_pickle_filepath = save_woe_pickle_filepath
        self.load_woe_pickle_filepath = load_woe_pickle_filepath
        self.min_sample_weight = min_sample_weight

    @property
    def columns(self):
        conf = pd.read_csv(self.config_filepath)
        return conf['var_name'].tolist()

    def fit(self, X, y):
        """Fit the WOE model on the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        self._prepare_data(X, y, FIT_DATA_PATH)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                fp.process_train_woe(
                    FIT_DATA_PATH,
                    FEATURE_DETAILS_OUTPUT,
                    self.save_woe_pickle_filepath,
                    self.config_filepath,
                    self.min_sample_weight)

        self.load_woe_pickle_filepath = self.save_woe_pickle_filepath

    def transform(self, X, y=None):
        """Transform the X features into WOE features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        if self.config_filepath is None:
            raise ValueError("Config filepath does not exist.\
                Please define a config file first.")

        if self.load_woe_pickle_filepath is None:
            raise ValueError("Load WOE pickle filepath does not exist.\
                Either fit model or load model first.")

        self._prepare_data(X, y, TRANSFORM_DATA_PATH)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                fp.process_woe_trans(
                    TRANSFORM_DATA_PATH,
                    self.load_woe_pickle_filepath,
                    TRANSFORMED_WOE_DATA_PATH,
                    self.config_filepath)

        df_processed = self._load_data(TRANSFORMED_WOE_DATA_PATH)
        X_processed = df_processed.drop('target', axis=1)
        y_processed = df_processed.target.values
        return X_processed, y_processed

    def _prepare_data(self, X, y, outpath):
        # Create directory if not exist
        subprocess.check_output('mkdir -p ./woe_processing', shell=True)
        data = X.copy()
        data['target'] = y
        data.to_csv(outpath, index=False)

    def _load_data(self, filepath):
        return pd.read_csv(TRANSFORMED_WOE_DATA_PATH)

    def _read_config_vars(self):
        conf = pd.read_csv(self.config_filepath)
        return conf.var_name.tolist()

    @staticmethod
    def create_new_config(config_filepath, df):
        n = df.shape[1]
        var_name = df.columns.tolist()
        var_dtype = [dt.name for dt in df.dtypes]
        is_tobe_bin = [0] * n
        is_candidate = [0] * n
        is_modelfeature = [0] * n
        pd.DataFrame({
            'var_name': var_name,
            'var_dtype': var_dtype,
            'is_tobe_bin': is_tobe_bin,
            'is_candidate': is_candidate,
            'is_modelfeature': is_modelfeature
        }).to_csv(config_filepath, index=False)

    @staticmethod
    def mark_woe_features(config_filepath, features):
        conf = pd.read_csv(config_filepath)
        mark_cols = ['is_tobe_bin', 'is_candidate', 'is_modelfeature']
        conf.loc[conf.var_name.isin(features), mark_cols] = 1
        conf.to_csv(config_filepath, index=False)
