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
from typing import Dict, Union

from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn, params

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST], provider="fate")
def feature_imputation(ctx, role):
    ...


@feature_imputation.train()
def feature_imputation_train(
        ctx: Context,
        role: Role,
        train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        method: cpn.parameter(type=params.string_choice(["min", "max", "mean", "random", "consts",
                                                         "none"]),
                              default="random", optional=True,
                              desc="imputation method, "
                                   "choose from {'min', 'max', 'mean', 'random', 'consts', 'none'},"
                                   "'none' means no imputation, default is 'random'"),
        col_fill_method: cpn.parameter(type=Dict[str, str], default=None, optional=True,
                                       desc="fill method for each column, "
                                            "any column unspecified will be filled by values computed with `method`"),
        fill_const: cpn.parameter(type=Union[float, int], default=0.0, optional=True,
                                  desc="fill constant for `consts` method"),
        missing_val: cpn.parameter(type=list, default=None, optional=True, desc="values to be treated as missing"),
        use_anonymous: cpn.parameter(
            type=bool, default=False, desc="bool, whether interpret keys in `col_fill_method` as anonymous column names"
        ),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST]),
):
    train(
        ctx,
        train_data,
        train_output_data,
        output_model,
        method,
        col_fill_method,
        fill_const,
        missing_val,
        use_anonymous,
    )


@feature_imputation.predict()
def feature_imputation_predict(
        ctx: Context,
        role: Role,
        test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        input_model: cpn.json_model_input(roles=[GUEST, HOST]),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    predict(ctx, input_model, test_data, test_output_data)


def train(
        ctx,
        train_data,
        train_output_data,
        output_model,
        method,
        col_fill_method,
        fill_const,
        missing_val,
        use_anonymous,
):
    logger.info(f"start imputation train")
    from fate.ml.preprocessing import FeatureImputation

    train_data = train_data.read()

    sub_ctx = ctx.sub_ctx("train")
    columns = train_data.schema.columns.to_list()
    anonymous_columns = None
    if use_anonymous:
        logger.info(f"use anonymous columns")
        anonymous_columns = train_data.schema.anonymous_columns.to_list()
    col_imputation_method_map = get_to_imputation_cols(columns, anonymous_columns, method, col_fill_method)

    imputation = FeatureImputation(col_imputation_method_map, fill_const, missing_val)
    imputation.fit(sub_ctx, train_data)

    model = imputation.get_model()
    output_model.write(model, metadata={})

    sub_ctx = ctx.sub_ctx("predict")
    output_data = imputation.transform(sub_ctx, train_data)
    logger.info(f"imputation output data: {output_data.as_pd_df().head(30)}")
    train_output_data.write(output_data)


def predict(ctx, input_model, test_data, test_output_data):
    logger.info(f"start imputation transform")

    from fate.ml.preprocessing import FeatureImputation

    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    imputation = FeatureImputation.from_model(model)
    test_data = test_data.read()
    output_data = imputation.transform(sub_ctx, test_data)
    logger.info(f"imputation output data: {output_data.as_pd_df().head(30)}")
    test_output_data.write(output_data)


def get_to_imputation_cols(columns, anonymous_columns, method, col_fill_method):
    if anonymous_columns is not None:
        col_name_map = dict(zip(anonymous_columns, columns))
    else:
        col_name_map = dict(zip(columns, columns))

    col_imputation_method_map = {}
    if col_fill_method is not None:
        for col in col_name_map:
            col_imputation_method_map[col_name_map[col]] = col_fill_method.get(col, method)
    else:
        col_imputation_method_map = {col: col_fill_method for col in col_name_map.values()}
    return col_imputation_method_map
