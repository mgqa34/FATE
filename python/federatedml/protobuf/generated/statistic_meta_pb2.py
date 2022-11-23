# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: statistic-meta.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='statistic-meta.proto',
    package='com.webank.ai.fate.core.mlmodel.buffer',
    syntax='proto3',
    serialized_options=b'B\022StatisticMetaProto',
    serialized_pb=b'\n\x14statistic-meta.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"e\n\rStatisticMeta\x12\x12\n\nstatistics\x18\x01 \x03(\t\x12\x16\n\x0estatic_columns\x18\x02 \x03(\t\x12\x16\n\x0equantile_error\x18\x03 \x01(\x01\x12\x10\n\x08need_run\x18\x04 \x01(\x08\x42\x14\x42\x12StatisticMetaProtob\x06proto3'
)


_STATISTICMETA = _descriptor.Descriptor(
    name='StatisticMeta',
    full_name='com.webank.ai.fate.core.mlmodel.buffer.StatisticMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='statistics', full_name='com.webank.ai.fate.core.mlmodel.buffer.StatisticMeta.statistics', index=0,
            number=1, type=9, cpp_type=9, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='static_columns', full_name='com.webank.ai.fate.core.mlmodel.buffer.StatisticMeta.static_columns', index=1,
            number=2, type=9, cpp_type=9, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='quantile_error', full_name='com.webank.ai.fate.core.mlmodel.buffer.StatisticMeta.quantile_error', index=2,
            number=3, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='need_run', full_name='com.webank.ai.fate.core.mlmodel.buffer.StatisticMeta.need_run', index=3,
            number=4, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=64,
    serialized_end=165,
)

DESCRIPTOR.message_types_by_name['StatisticMeta'] = _STATISTICMETA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StatisticMeta = _reflection.GeneratedProtocolMessageType('StatisticMeta', (_message.Message,), {
    'DESCRIPTOR': _STATISTICMETA,
    '__module__': 'statistic_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.StatisticMeta)
})
_sym_db.RegisterMessage(StatisticMeta)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)