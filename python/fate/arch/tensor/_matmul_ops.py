from ._base import DStorage, Shape
from ._ops import _get_dispatch_info, dispatch_signature2
from ._storage_ops import _ops_dispatch_signature2_local_unknown_unknown
from ._tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    If both arguments are 2-D they are multiplied like conventional matrices.
    If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
    If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
    """
    _is_distributed, _device, _dtype = _get_dispatch_info([a, b])

    # both local
    if not _is_distributed:
        storage_op = _ops_dispatch_signature2_local_unknown_unknown(
            "matmul", _device, _dtype, [], {}
        )
        storage = storage_op(a.storage, b.storage)
        return Tensor(storage)

    bc_shape_a = a.shape[:-2]
    bc_shape_b = b.shape[:-2]
    bs_shape = Shape.broadcast_shape([bc_shape_a, bc_shape_b], raise_exception=False)
    if bs_shape is None:
        raise ValueError("matmul: shape broadcast failed")

    if bc_shape_a.d_axis is not None:
        # distributed along bc part: (...,d,...,m, k) x (...,d,...,k, n) -> (...,d,..., m, n)
        # join and matmul
        return dispatch_signature2("matmul", a, b, [], {}, bc_shape_validate=False)

    mul_shape_a = a.shape[-2:]
    mul_shape_b = b.shape[-2:]
    if mul_shape_a.size[-1] != mul_shape_b.size[0]:
        raise ValueError("matmul: dimension mismatch: should be (..., n) x (...,n,?)")

    if mul_shape_a.is_d_axis(-2) or mul_shape_b.is_d_axis(-1):
        raise ValueError("not supported distributed axis position")

    out_storage = a.storage.blocks.join(
        b.storage.blocks,
        _ops_dispatch_signature2_local_unknown_unknown(
            "matmul", _device, _dtype, [], {}
        ),
    ).reduce(
        _ops_dispatch_signature2_local_unknown_unknown("add", _device, _dtype, [], {})
    )
    return Tensor(out_storage)