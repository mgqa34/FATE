# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data-io-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13\x64\x61ta-io-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"\xdc\x02\n\x0cImputerParam\x12l\n\x15missing_replace_value\x18\x01 \x03(\x0b\x32M.com.webank.ai.fate.core.mlmodel.buffer.ImputerParam.MissingReplaceValueEntry\x12h\n\x13missing_value_ratio\x18\x02 \x03(\x0b\x32K.com.webank.ai.fate.core.mlmodel.buffer.ImputerParam.MissingValueRatioEntry\x1a:\n\x18MissingReplaceValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x38\n\x16MissingValueRatioEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\"\xdc\x02\n\x0cOutlierParam\x12l\n\x15outlier_replace_value\x18\x01 \x03(\x0b\x32M.com.webank.ai.fate.core.mlmodel.buffer.OutlierParam.OutlierReplaceValueEntry\x12h\n\x13outlier_value_ratio\x18\x02 \x03(\x0b\x32K.com.webank.ai.fate.core.mlmodel.buffer.OutlierParam.OutlierValueRatioEntry\x1a:\n\x18OutlierReplaceValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x38\n\x16OutlierValueRatioEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\"\xdd\x01\n\x0b\x44\x61taIOParam\x12\x0e\n\x06header\x18\x01 \x03(\t\x12\x10\n\x08sid_name\x18\x02 \x01(\t\x12\x12\n\nlabel_name\x18\x03 \x01(\t\x12K\n\rimputer_param\x18\x04 \x01(\x0b\x32\x34.com.webank.ai.fate.core.mlmodel.buffer.ImputerParam\x12K\n\routlier_param\x18\x05 \x01(\x0b\x32\x34.com.webank.ai.fate.core.mlmodel.buffer.OutlierParamB\x12\x42\x10\x44\x61taIOParamProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'data_io_param_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\020DataIOParamProto'
  _IMPUTERPARAM_MISSINGREPLACEVALUEENTRY._options = None
  _IMPUTERPARAM_MISSINGREPLACEVALUEENTRY._serialized_options = b'8\001'
  _IMPUTERPARAM_MISSINGVALUERATIOENTRY._options = None
  _IMPUTERPARAM_MISSINGVALUERATIOENTRY._serialized_options = b'8\001'
  _OUTLIERPARAM_OUTLIERREPLACEVALUEENTRY._options = None
  _OUTLIERPARAM_OUTLIERREPLACEVALUEENTRY._serialized_options = b'8\001'
  _OUTLIERPARAM_OUTLIERVALUERATIOENTRY._options = None
  _OUTLIERPARAM_OUTLIERVALUERATIOENTRY._serialized_options = b'8\001'
  _globals['_IMPUTERPARAM']._serialized_start=64
  _globals['_IMPUTERPARAM']._serialized_end=412
  _globals['_IMPUTERPARAM_MISSINGREPLACEVALUEENTRY']._serialized_start=296
  _globals['_IMPUTERPARAM_MISSINGREPLACEVALUEENTRY']._serialized_end=354
  _globals['_IMPUTERPARAM_MISSINGVALUERATIOENTRY']._serialized_start=356
  _globals['_IMPUTERPARAM_MISSINGVALUERATIOENTRY']._serialized_end=412
  _globals['_OUTLIERPARAM']._serialized_start=415
  _globals['_OUTLIERPARAM']._serialized_end=763
  _globals['_OUTLIERPARAM_OUTLIERREPLACEVALUEENTRY']._serialized_start=647
  _globals['_OUTLIERPARAM_OUTLIERREPLACEVALUEENTRY']._serialized_end=705
  _globals['_OUTLIERPARAM_OUTLIERVALUERATIOENTRY']._serialized_start=707
  _globals['_OUTLIERPARAM_OUTLIERVALUERATIOENTRY']._serialized_end=763
  _globals['_DATAIOPARAM']._serialized_start=766
  _globals['_DATAIOPARAM']._serialized_end=987
# @@protoc_insertion_point(module_scope)
