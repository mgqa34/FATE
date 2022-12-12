from contextlib import contextmanager
from copy import copy
from typing import Iterator, List, Optional

from fate.interface import T_ROLE, ComputingEngine
from fate.interface import Context as ContextInterface
from fate.interface import FederationEngine, PartyMeta

from ..unify import device
from ._cipher import CipherKit
from ._federation import GC, Parties, Party
from ._io import IOKit
from ._metrics import MetricsHandler, NoopMetricsHandler
from ._namespace import Namespace
from ._tensor import TensorKit


class Context(ContextInterface):
    """
    implement fate.interface.ContextInterface

    Note: most parameters has default dummy value,
          which is convenient when used in script.
          please pass in custom implements as you wish
    """

    def __init__(
        self,
        context_name: Optional[str] = None,
        device: device = device.CPU,
        computing: Optional[ComputingEngine] = None,
        federation: Optional[FederationEngine] = None,
        metrics: MetricsHandler = NoopMetricsHandler(),
        namespace: Optional[Namespace] = None,
    ) -> None:
        self.context_name = context_name
        self.metrics = metrics

        if namespace is None:
            namespace = Namespace()
        self.namespace = namespace

        self.cipher: CipherKit = CipherKit(device)
        self.tensor: TensorKit = TensorKit(computing, device)
        self._io_kit: IOKit = IOKit()

        self._computing = computing
        self._federation = federation
        self._role_to_parties = None

        self._gc = GC()

    def with_namespace(self, namespace: Namespace):
        context = copy(self)
        context.namespace = namespace
        return context

    def range(self, end):
        for i in range(end):
            yield i, self.with_namespace(self.namespace.sub_namespace(f"{i}"))

    def iter(self, iterable):
        for i, it in enumerate(iterable):
            yield self.with_namespace(self.namespace.sub_namespace(f"{i}")), it

    @property
    def computing(self):
        return self._get_computing()

    @property
    def federation(self):
        return self._get_federation()

    @contextmanager
    def sub_ctx(self, namespace: str) -> Iterator["Context"]:
        try:
            yield self.with_namespace(self.namespace.sub_namespace(namespace))
        finally:
            ...

    def set_federation(self, federation: FederationEngine):
        self._federation = federation

    @property
    def guest(self) -> Party:
        return Party(
            self._get_federation(),
            self._get_parties("guest")[0],
            self.namespace,
        )

    @property
    def hosts(self) -> Parties:
        return Parties(
            self._get_federation(),
            self._get_federation().local_party,
            self._get_parties("host"),
            self.namespace,
        )

    @property
    def arbiter(self) -> Party:
        return Party(
            self._get_federation(),
            self._get_parties("arbiter")[0],
            self.namespace,
        )

    @property
    def local(self):
        return self._get_federation().local_party

    @property
    def parties(self) -> Parties:
        return Parties(
            self._get_federation(),
            self._get_federation().local_party,
            self._get_parties(),
            self.namespace,
        )

    def _get_parties(self, role: Optional[T_ROLE] = None) -> List[PartyMeta]:
        # update role to parties mapping
        if self._role_to_parties is None:
            self._role_to_parties = {}
            for party in self._get_federation().parties:
                self._role_to_parties.setdefault(party[0], []).append(party)

        parties = []
        if role is None:
            for role_parties in self._role_to_parties.values():
                parties.extend(role_parties)
        else:
            if role not in self._role_to_parties:
                raise RuntimeError(f"no {role} party has configurated")
            else:
                parties.extend(self._role_to_parties[role])
        return parties

    def _get_federation(self):
        if self._federation is None:
            raise RuntimeError(f"federation not set")
        return self._federation

    def _get_computing(self):
        if self._computing is None:
            raise RuntimeError(f"computing not set")
        return self._computing

    def reader(self, uri, **kwargs) -> "Reader":
        return self._io_kit.reader(self, uri, **kwargs)

    def writer(self, uri, **kwargs) -> "Writer":
        return self._io_kit.writer(self, uri, **kwargs)
