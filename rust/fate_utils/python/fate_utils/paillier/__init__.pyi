from typing import List, Tuple, Union

class PK:
    def __new__(self) -> PK: ...
    def __getstate__(self) -> List[bytes]: ...
    def __setstate__(self, state: List[bytes]) -> None: ...

class SK:
    def __new__(self) -> SK: ...
    def __getstate__(self) -> List[bytes]: ...
    def __setstate__(self, state: List[bytes]) -> None: ...

class Coder:
    def __new__(self) -> Coder: ...
    def __getstate__(self) -> List[bytes]: ...
    def __setstate__(self, state: List[bytes]) -> None: ...

    @staticmethod
    def from_pk(pk: PK) -> Coder: ...
    def encode_f64(self, data: float) -> Plaintext: ...
    def decode_f64(self, data: Plaintext) -> float: ...
    def encode_f32(self, data: float) -> Plaintext: ...
    def decode_f32(self, data: Plaintext) -> float: ...
    # ... [Similar methods for i64, i32, etc.]

class Ciphertext: ...

class CiphertextVector:
    def __new__(self) -> CiphertextVector: ...
    def __getstate__(self) -> List[bytes]: ...
    def __setstate__(self, state: List[bytes]) -> None: ...
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    def slice_indexes(self, indexes: List[int]) -> CiphertextVector: ...

class Plaintext: ...

class PlaintextVector:
    def __new__(self) -> PlaintextVector: ...
    def __getstate__(self) -> List[bytes]: ...
    def __setstate__(self, state: List[bytes]) -> None: ...
    def __str__(self) -> str: ...

class Evaluator:
    @staticmethod
    def encrypt(pk: PK, plaintext_vector: PlaintextVector, obfuscate: bool) -> CiphertextVector: ...

    @staticmethod
    def encrypt_scalar(pk: PK, plaintext: Plaintext, obfuscate: bool) -> Ciphertext: ...

    @staticmethod
    def decrypt(sk: SK, data: CiphertextVector) -> PlaintextVector: ...

    @staticmethod
    def decrypt_scalar(sk: SK, data: Ciphertext) -> Plaintext: ...

    @staticmethod
    def zeros(size: int) -> CiphertextVector: ...

    @staticmethod
    def cat(vec_list: List[CiphertextVector]) -> CiphertextVector: ...

    @staticmethod
    def slice_indexes(a: CiphertextVector, indexes: List[int]) -> CiphertextVector: ...

    @staticmethod
    def slice(a: CiphertextVector, start: int, end: int) -> CiphertextVector: ...

    @staticmethod
    def reshape(a: CiphertextVector, shape: List[int]) -> CiphertextVector: ...

    @staticmethod
    def mul(pk: PK, a: CiphertextVector, b: CiphertextVector) -> CiphertextVector: ...

    @staticmethod
    def mul_plain(pk: PK, a: CiphertextVector, b: PlaintextVector) -> CiphertextVector: ...

    @staticmethod
    def add(a: CiphertextVector, b: CiphertextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def add_plain(a: CiphertextVector, b: PlaintextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def sub(a: CiphertextVector, b: CiphertextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def sub_plain(a: CiphertextVector, b: PlaintextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def intervals_sum_with_step(
            a: CiphertextVector, pk: PK, intervals: List[Tuple[int, int]], step: int
    ) -> CiphertextVector: ...

    @staticmethod
    def intervals_dot_with_step(
            a: CiphertextVector, pk: PK, b: CiphertextVector, intervals: List[Tuple[int, int]], step: int
    ) -> CiphertextVector: ...
class ArithmeticEvaluator:
    @staticmethod
    def zeros(size: int) -> CiphertextVector: ...

    @staticmethod
    def iadd(a: CiphertextVector, pk: PK, other: CiphertextVector) -> None: ...

    @staticmethod
    def idouble(a: CiphertextVector, pk: PK) -> None: ...

    @staticmethod
    def add(a: CiphertextVector, other: CiphertextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def add_scalar(a: CiphertextVector, other: Ciphertext, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def sub(a: CiphertextVector, other: CiphertextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def sub_scalar(a: CiphertextVector, other: Ciphertext, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def rsub(a: CiphertextVector, other: CiphertextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def rsub_scalar(a: CiphertextVector, other: Ciphertext, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def mul(a: CiphertextVector, other: PlaintextVector, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def mul_scalar(a: CiphertextVector, other: Plaintext, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def matmul(a: CiphertextVector, other: PlaintextVector, lshape: List[int], rshape: List[int], pk: PK) -> CiphertextVector: ...

    @staticmethod
    def rmatmul(a: CiphertextVector, other: PlaintextVector, lshape: List[int], rshape: List[int], pk: PK) -> CiphertextVector: ...

class BaseEvaluator:
    @staticmethod
    def encrypt(plaintext_vector: PlaintextVector, obfuscate: bool, pk: PK) -> CiphertextVector: ...

    @staticmethod
    def encrypt_scalar(plaintext: Plaintext, obfuscate: bool, pk: PK) -> Ciphertext: ...

    @staticmethod
    def decrypt(data: CiphertextVector, sk: SK) -> PlaintextVector: ...

    @staticmethod
    def decrypt_scalar(data: Ciphertext, sk: SK) -> Plaintext: ...

    @staticmethod
    def cat(vec_list: List[CiphertextVector]) -> CiphertextVector: ...

class HistogramEvaluator:
    @staticmethod
    def slice_indexes(a: CiphertextVector, indexes: List[int]) -> CiphertextVector: ...

    @staticmethod
    def intervals_sum_with_step(
            a: CiphertextVector,
            intervals: List[Tuple[int, int]],
            step: int,
            pk: PK,
    ) -> CiphertextVector: ...

    @staticmethod
    def iupdate(
            a: CiphertextVector,
            b: CiphertextVector,
            indexes: List[List[int]],
            stride: int,
            pk: PK
    ) -> None: ...

    @staticmethod
    def iadd_vec(
            a: CiphertextVector,
            b: CiphertextVector,
            sa: int,
            sb: int,
            size: Optional[int],
            pk: PK
    ) -> None: ...

    @staticmethod
    def chunking_cumsum_with_step(
            a: CiphertextVector,
            chunk_sizes: List[int],
            step: int,
            pk: PK,
    ) -> None: ...

    @staticmethod
    def tolist(a: CiphertextVector) -> List[Ciphertext]: ...

    @staticmethod
    def pack_squeeze(
            a: CiphertextVector,
            pack_num: int,
            offset_bit: int,
            pk: PK
    ) -> CiphertextVector: ...

    @staticmethod
    def slice(a: CiphertextVector, start: int, size: int) -> CiphertextVector: ...

    @staticmethod
    def i_shuffle(a: CiphertextVector, indexes: List[int]) -> None: ...

    @staticmethod
    def intervals_slice(
            a: CiphertextVector,
            intervals: List[Tuple[int, int]]
    ) -> CiphertextVector: ...

    @staticmethod
    def iadd_slice(
            a: CiphertextVector,
            position: int,
            other: List[Ciphertext],
            pk: PK
    ) -> None: ...

    @staticmethod
    def iadd_vec_self(
            a: CiphertextVector,
            sa: int,
            sb: int,
            size: Optional[int],
            pk: PK
    ) -> None: ...

def keygen(bit_length: int) -> Tuple[SK, PK]: ...
