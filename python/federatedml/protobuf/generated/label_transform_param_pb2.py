# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: label-transform-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1blabel-transform-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"\xfa\x03\n\x13LabelTransformParam\x12\x64\n\rlabel_encoder\x18\x01 \x03(\x0b\x32M.com.webank.ai.fate.core.mlmodel.buffer.LabelTransformParam.LabelEncoderEntry\x12i\n\x10\x65ncoder_key_type\x18\x02 \x03(\x0b\x32O.com.webank.ai.fate.core.mlmodel.buffer.LabelTransformParam.EncoderKeyTypeEntry\x12m\n\x12\x65ncoder_value_type\x18\x03 \x03(\x0b\x32Q.com.webank.ai.fate.core.mlmodel.buffer.LabelTransformParam.EncoderValueTypeEntry\x1a\x33\n\x11LabelEncoderEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x35\n\x13\x45ncoderKeyTypeEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x37\n\x15\x45ncoderValueTypeEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42\x1a\x42\x18LabelTransformParamProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'label_transform_param_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\030LabelTransformParamProto'
  _LABELTRANSFORMPARAM_LABELENCODERENTRY._options = None
  _LABELTRANSFORMPARAM_LABELENCODERENTRY._serialized_options = b'8\001'
  _LABELTRANSFORMPARAM_ENCODERKEYTYPEENTRY._options = None
  _LABELTRANSFORMPARAM_ENCODERKEYTYPEENTRY._serialized_options = b'8\001'
  _LABELTRANSFORMPARAM_ENCODERVALUETYPEENTRY._options = None
  _LABELTRANSFORMPARAM_ENCODERVALUETYPEENTRY._serialized_options = b'8\001'
  _globals['_LABELTRANSFORMPARAM']._serialized_start=72
  _globals['_LABELTRANSFORMPARAM']._serialized_end=578
  _globals['_LABELTRANSFORMPARAM_LABELENCODERENTRY']._serialized_start=415
  _globals['_LABELTRANSFORMPARAM_LABELENCODERENTRY']._serialized_end=466
  _globals['_LABELTRANSFORMPARAM_ENCODERKEYTYPEENTRY']._serialized_start=468
  _globals['_LABELTRANSFORMPARAM_ENCODERKEYTYPEENTRY']._serialized_end=521
  _globals['_LABELTRANSFORMPARAM_ENCODERVALUETYPEENTRY']._serialized_start=523
  _globals['_LABELTRANSFORMPARAM_ENCODERVALUETYPEENTRY']._serialized_end=578
# @@protoc_insertion_point(module_scope)
