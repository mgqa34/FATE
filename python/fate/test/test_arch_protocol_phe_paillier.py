import torch
from fate.arch.protocol.phe.paillier import keygen, Coder, evaluator


def test_pack_float():
    offset_bit = 32
    precision = 16
    vec = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    packed = Coder.pack_floats(vec, offset_bit, 2, precision)
    unpacked = Coder.unpack_floats(packed, offset_bit, 2, precision, 5)
    assert torch.allclose(vec, unpacked, rtol=1e-3, atol=1e-3)


def test_pack_squeeze():
    offset_bit = 32
    precision = 16
    pack_num = 2
    pack_packed_num = 2
    vec1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    vec2 = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])
    sk, pk = keygen(1024)
    coder = Coder.from_pk(pk)
    a = coder.pack_floats(vec1, offset_bit, pack_num, precision)
    ea = pk.encrypt_encoded(a, obfuscate=False)
    b = coder.pack_floats(vec2, offset_bit, pack_num, precision)
    eb = pk.encrypt_encoded(b, obfuscate=False)
    ec = evaluator.add(ea, eb, pk)

    # pack packed encrypted
    ec_pack = evaluator.pack_squeeze(ec, pack_packed_num, offset_bit * 2, pk)
    c_pack = sk.decrypt_to_encoded(ec_pack)
    c = coder.unpack_floats(c_pack, offset_bit, pack_num * pack_packed_num, precision, 5)
    assert torch.allclose(vec1 + vec2, c, rtol=1e-3, atol=1e-3)


def test_sum():
    sk, pk = keygen(1024)
    coder = Coder.from_pk(pk)
    vec = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    encrypted = pk.encrypt_encoded(coder.encode_tensor(vec), obfuscate=False)
    sum = evaluator.sum(encrypted, [2, 3], None, pk)
    sum = sk.decrypt_to_encoded(sum)
    sum = coder.decode_tensor(sum, dtype=vec.dtype)
    print(sum)
