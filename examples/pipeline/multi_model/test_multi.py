#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
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
import argparse

from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import PSI, HeteroFeatureSelection, HeteroFeatureBinning, \
    FeatureScale, Union, DataSplit, CoordinatedLR, CoordinatedLinR, Statistics, Sample, Evaluation, Reader
from fate_client.pipeline.utils import test_utils


def main(config="../config.yaml", namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host, arbiter=arbiter)
    if config.task_cores:
        pipeline.conf.set("task_cores", config.task_cores)
    if config.timeout:
        pipeline.conf.set("timeout", config.timeout)

    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
    reader_0.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_guest"
    )
    reader_0.hosts[0].task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_host"
    )
    psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])

    data_split_0 = DataSplit("data_split_0", input_data=psi_0.outputs["output_data"],
                             train_size=0.8, test_size=0.2, random_state=42)
    union_0 = Union("union_0", input_datas=[data_split_0.outputs["train_output_data"],
                                            data_split_0.outputs["test_output_data"]])
    sample_0 = Sample("sample_0", input_data=data_split_0.outputs["train_output_data"],
                      n=800, replace=True, hetero_sync=True)

    binning_0 = HeteroFeatureBinning("binning_0",
                                     method="quantile",
                                     n_bins=10,
                                     train_data=sample_0.outputs["output_data"]
                                     )
    statistics_0 = Statistics("statistics_0",
                              input_data=psi_0.outputs["output_data"])
    selection_0 = HeteroFeatureSelection("selection_0",
                                         method=["iv", "statistics"],
                                         train_data=sample_0.outputs["output_data"],
                                         input_models=[binning_0.outputs["output_model"],
                                                       statistics_0.outputs["output_model"]],
                                         iv_param={"metrics": "iv", "filter_type": "threshold", "value": 0.1},
                                         statistic_param={"metrics": ["max", "min"], "filter_type": "top_k",
                                                          "threshold": 5})

    selection_1 = HeteroFeatureSelection("selection_1",
                                         input_model=selection_0.outputs["train_output_model"],
                                         test_data=data_split_0.outputs["test_output_data"])

    scale_0 = FeatureScale("scale_0", method="min_max",
                           train_data=selection_0.outputs["train_output_data"], )

    lr_0 = CoordinatedLR("lr_0", train_data=selection_0.outputs["train_output_data"],
                         validate_data=selection_1.outputs["test_output_data"], epochs=3)
    linr_0 = CoordinatedLinR("linr_0", train_data=selection_0.outputs["train_output_data"],
                             validate_data=selection_1.outputs["test_output_data"], epochs=3)

    evaluation_0 = Evaluation("evaluation_0", input_datas=lr_0.outputs["train_output_data"],
                              default_eval_setting="binary",
                              runtime_parties=dict(guest=guest))
    evaluation_1 = Evaluation("evaluation_1", input_datas=linr_0.outputs["train_output_data"],
                              default_eval_setting="regression",
                              runtime_parties=dict(guest=guest))
    pipeline.add_tasks([reader_0, psi_0, data_split_0, union_0, sample_0, binning_0, statistics_0, selection_0,
                       scale_0, selection_1, lr_0, linr_0, evaluation_0, evaluation_1])

    # pipeline.add_task(hetero_feature_binning_0)
    pipeline.compile()
    # print(pipeline.get_dag())
    pipeline.fit()

    # print(pipeline.get_task_info("feature_scale_1").get_output_model())

    pipeline.deploy([psi_0, selection_0])

    predict_pipeline = FateFlowPipeline()
    reader_1 = Reader("reader_1", runtime_parties=dict(guest=guest, host=host))
    reader_1.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_guest"
    )
    reader_1.hosts[0].task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_host"
    )
    deployed_pipeline = pipeline.get_deployed_pipeline()
    deployed_pipeline.psi_0.input_data = reader_1.outputs["output_data"]

    predict_pipeline.add_tasks([reader_1, deployed_pipeline])
    predict_pipeline.compile()
    predict_pipeline.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
