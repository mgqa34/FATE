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

from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn, params


@cpn.component(roles=[GUEST, HOST])
def feature_union(ctx, role):
    ...


@feature_union.train()
def union_train(
        ctx: Context,
        role: Role,
        train_data_list: cpn.dataframe_inputs(roles=[GUEST, HOST]),
        axis: cpn.parameter(type=params.conint(strict=True, ge=0, le=1), default=0, optional=False,
                            desc="axis along which concatenation is performed, 0 for row-wise, 1 for column-wise"),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST])
):
    train(ctx, train_data_list, train_output_data, axis)


def train(ctx, train_data_list, train_output_data, axis):
    from fate.ml.preprocessing import FeatureUnion
    data_list = []
    for data in train_data_list:
        data = data.read()
        data_list.append(data)

    sub_ctx = ctx.sub_ctx("train")
    union_obj = FeatureUnion(axis)
    output_data = union_obj.fit(sub_ctx, data_list)
    train_output_data.write(output_data)