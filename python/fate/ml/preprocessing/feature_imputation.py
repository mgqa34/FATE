#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging

import pandas as pd

from fate.arch import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureImputation(Module):
    def __init__(self, col_imputation_method_map, fill_const=None, missing_val=None):
        self.col_imputation_method_map = col_imputation_method_map
        self.fill_const = fill_const
        self.missing_val = missing_val
        self._method_col_map = None
        self._mean = None
        self._max = None
        self._min = None
        self._consts = None
        self._skip_col = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        self._method_col_map = {}
        for col, method in self.col_imputation_method_map.items():
            if col not in train_data.schema.columns:
                raise ValueError(f"column {col} not in train data")
            if self._method_col_map.get(method) is None:
                self._method_col_map[method] = []
            self._method_col_map[method].append(col)
        for method, cols in self._method_col_map.items():
            if method == "max":
                self._max = train_data[cols].max()
            elif method == "min":
                self._min = train_data[cols].min()
            elif method == "mean":
                self._mean = train_data[cols].mean()
            elif method == "const":
                self._consts = pd.Series(self.fill_const, index=cols)
            elif method == "none":
                self._skip_col = cols

    def transform(self, ctx: Context, test_data):
        to_compute_data = test_data
        if self._skip_col is not None:
            to_compute_col = [col for col in test_data.schema.columns if col not in self._skip_col]
            if len(to_compute_col) == 0:
                return test_data
            to_compute_data = test_data[to_compute_col]
        logger.info(f"to_compute_data: {to_compute_data.as_pd_df().head(5)}")
        if self.missing_val:
            # @todo: need df nan support
            for missing_val in self.missing_val:
                logger.info(f"missing value: {missing_val}")
                to_compute_data = to_compute_data[to_compute_data != missing_val]
                logger.info(f"to_compute_data: {to_compute_data.as_pd_df()}")
        for method, cols in self._method_col_map.items():
            if method == "max":
                test_data[cols] = to_compute_data[cols].fillna(self._max)
            elif method == "min":
                test_data[cols] = to_compute_data[cols].fillna(self._min)
            elif method == "mean":
                test_data[cols] = to_compute_data[cols].fillna(self._mean)
            elif method == "const":
                test_data[cols] = to_compute_data[cols].fillna(self._consts)
            elif method == "random":
                # @todo: fill by random value per entry
                pass
                # fill_val = torch.randn(len(cols))
                # fill_df = torch.randn(to_compute_df.shape)
                # test_data[cols] = to_compute_data[cols].fillna(fill_df)
                # test_data[cols] = to_compute_data[cols].apply(lambda x: random.random() if x is None else x)
        return test_data

    def get_model(self):
        mean_dict = self._mean.to_dict() if self._mean is not None else None
        max_dict = self._max.to_dict() if self._max is not None else None
        min_dict = self._min.to_dict() if self._min is not None else None
        consts_dict = self._consts.to_dict() if self._consts is not None else None
        model_data = {"method_col_map": self._method_col_map,
                      "mean": mean_dict,
                      "max": max_dict,
                      "min": min_dict,
                      "consts": consts_dict,
                      "skip_col": self._skip_col}
        return {"data": model_data, "meta": {"col_imputation_method_map": self.col_imputation_method_map,
                                             "fill_const": self.fill_const,
                                             "missing_val": self.missing_val,
                                             "model_type": "feature_imputation"}}

    def restore(self, model):
        self._method_col_map = model["method_col_map"]
        self._mean = pd.Series(model["mean"]) if model["mean"] is not None else None
        self._max = pd.Series(model["max"]) if model["max"] is not None else None
        self._min = pd.Series(model["min"]) if model["min"] is not None else None
        self._consts = pd.Series(model["consts"]) if model["consts"] is not None else None
        self._skip_col = model["skip_col"]

    @classmethod
    def from_model(cls, model) -> "FeatureImputation":
        imputation_tool = FeatureImputation(model["meta"]["col_imputation_method_map"],
                                            model["meta"]["fill_const"],
                                            model["meta"]["missing_val"])
        imputation_tool.restore(model["data"])
        return imputation_tool
