from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic
from fate.components.spec.mlmd import FlowMLMD, PipelineMLMD

from .logger import CustomLogger, FlowLogger, PipelineLogger


class TaskExecuteStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


class TaskDistributedComputingBackendSpec(pydantic.BaseModel):
    engine: str
    computing_id: str


class TaskPartySpec(pydantic.BaseModel):
    role: Literal["guest", "host", "arbiter"]
    partyid: str


class TaskFederationPartiesSpec(pydantic.BaseModel):
    local: TaskPartySpec
    parties: List[TaskPartySpec]


class TaskFederationBackendSpec(pydantic.BaseModel):
    engine: str
    federation_id: str
    parties: TaskFederationPartiesSpec


class TaskEnvSpec(pydantic.BaseModel):
    device: Literal["CPU", "GPU"]
    distributed_computing_backend: TaskDistributedComputingBackendSpec
    federation_backend: TaskFederationBackendSpec
    logger: Union[PipelineLogger, FlowLogger, CustomLogger]
    mlmd: Union[PipelineMLMD, FlowMLMD]

    def get_device(self):
        from fate.arch.unify import device

        for dev in device:
            if dev.name == self.device.strip().upper():
                return dev
        raise ValueError(f"should be one of {[dev.name for dev in device]}")


class ArtifactSpec(pydantic.BaseModel):
    name: str
    uri: str
    metadata: Optional[dict] = None


class TaskInputsSpec(pydantic.BaseModel):
    parameters: Dict[str, Any]
    artifacts: Dict[str, Union[ArtifactSpec, List[ArtifactSpec]]]


class TaskOutputsSpec(pydantic.BaseModel):
    artifacts: Dict[str, Union[ArtifactSpec, List[ArtifactSpec]]]


class TaskConfigSpec(pydantic.BaseModel):
    execution_id: str
    component: str
    role: str
    stage: str
    inputs: TaskInputsSpec
    outputs: TaskOutputsSpec
    env: TaskEnvSpec