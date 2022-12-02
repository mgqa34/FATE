from .runtime_entity import PartySpec
from .dag_structures import RuntimeInputDefinition, DAGSpec, DAGSchema, \
    TaskSpec, PartyTaskRefSpec, PartyTaskSpec, JobConfSpec
from ..scheduler.component_stage import ComponentStageSchedule

SCHEMA_VERSION = "2.0.0.alpha"


class DAG(object):
    def __init__(self):
        self._dag_spec = None
        self._is_compiled = False

    @property
    def dag_spec(self):
        if not self._is_compiled:
            raise ValueError("Please compile pipeline first")

        return DAGSchema(dag=self._dag_spec, schema_version=SCHEMA_VERSION)

    def compile(self, roles, task_insts, stage, job_conf):
        scheduler_party_id = roles.scheduler_party_id
        parties = roles.get_parties_spec()
        tasks = dict()
        party_tasks = dict()
        for task_name, task_inst in task_insts.items():
            task = dict(component_ref=task_inst.component_ref)
            dependent_tasks = task_inst.get_dependent_tasks()
            inputs = RuntimeInputDefinition()
            input_channel, input_artifacts = task_inst.get_runtime_input_artifacts()
            task_stage = ComponentStageSchedule.get_stage(input_artifacts)
            if task_stage != stage:
                task["stage"] = task_stage

            if input_channel:
                inputs.artifacts = input_channel

            if dependent_tasks:
                task["dependent_tasks"] = dependent_tasks

            cpn_runtime_roles = set(roles.get_runtime_roles()) & set(task_inst.support_roles)
            if cpn_runtime_roles != set(roles.get_runtime_roles()):
                task["parties"] = roles.get_parties_spec(cpn_runtime_roles)

            common_parameters = task_inst.get_component_param()
            if common_parameters:
                inputs.parameters = common_parameters

            for role in cpn_runtime_roles:
                party_id_list = roles.get_party_id_list_by_role(role)
                for idx, party_id in enumerate(party_id_list):
                    role_param = task_inst.get_role_param(role, idx)
                    if role_param:
                        role_party_key = f"{role}_{party_id}"
                        if role_party_key not in party_tasks:
                            party_tasks[role_party_key] = PartyTaskSpec(
                                parties=[PartySpec(role=role, party_id=[party_id])],
                                tasks=dict()
                            )

                        party_tasks[role_party_key].tasks[task_name] = PartyTaskRefSpec(
                            inputs=RuntimeInputDefinition(
                                parameters=role_param
                            )
                        )

            if inputs.dict(exclude_unset=True):
                task["inputs"] = inputs

            tasks[task_name] = TaskSpec(**task)

        self._dag_spec = DAGSpec(
            scheduler_party_id=scheduler_party_id,
            parties=parties,
            stage=stage,
            tasks=tasks
        )
        if job_conf:
            self._dag_spec.conf = JobConfSpec(**job_conf)
        if party_tasks:
            self._dag_spec.party_tasks = party_tasks

        self._is_compiled = True
