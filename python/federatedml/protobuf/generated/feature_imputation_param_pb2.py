# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature-imputation-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1e\x66\x65\x61ture-imputation-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"\xed\x05\n\x13\x46\x65\x61tureImputerParam\x12s\n\x15missing_replace_value\x18\x01 \x03(\x0b\x32T.com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerParam.MissingReplaceValueEntry\x12o\n\x13missing_value_ratio\x18\x02 \x03(\x0b\x32R.com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerParam.MissingValueRatioEntry\x12|\n\x1amissing_replace_value_type\x18\x03 \x03(\x0b\x32X.com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerParam.MissingReplaceValueTypeEntry\x12\x11\n\tskip_cols\x18\x04 \x03(\t\x12o\n\x13\x63ols_replace_method\x18\x05 \x03(\x0b\x32R.com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerParam.ColsReplaceMethodEntry\x1a:\n\x18MissingReplaceValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x38\n\x16MissingValueRatioEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\x1a>\n\x1cMissingReplaceValueTypeEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x38\n\x16\x43olsReplaceMethodEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"|\n\x16\x46\x65\x61tureImputationParam\x12\x0e\n\x06header\x18\x01 \x03(\t\x12R\n\rimputer_param\x18\x02 \x01(\x0b\x32;.com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerParamB\x1d\x42\x1b\x46\x65\x61tureImputationParamProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'feature_imputation_param_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\033FeatureImputationParamProto'
  _FEATUREIMPUTERPARAM_MISSINGREPLACEVALUEENTRY._options = None
  _FEATUREIMPUTERPARAM_MISSINGREPLACEVALUEENTRY._serialized_options = b'8\001'
  _FEATUREIMPUTERPARAM_MISSINGVALUERATIOENTRY._options = None
  _FEATUREIMPUTERPARAM_MISSINGVALUERATIOENTRY._serialized_options = b'8\001'
  _FEATUREIMPUTERPARAM_MISSINGREPLACEVALUETYPEENTRY._options = None
  _FEATUREIMPUTERPARAM_MISSINGREPLACEVALUETYPEENTRY._serialized_options = b'8\001'
  _FEATUREIMPUTERPARAM_COLSREPLACEMETHODENTRY._options = None
  _FEATUREIMPUTERPARAM_COLSREPLACEMETHODENTRY._serialized_options = b'8\001'
  _globals['_FEATUREIMPUTERPARAM']._serialized_start=75
  _globals['_FEATUREIMPUTERPARAM']._serialized_end=824
  _globals['_FEATUREIMPUTERPARAM_MISSINGREPLACEVALUEENTRY']._serialized_start=586
  _globals['_FEATUREIMPUTERPARAM_MISSINGREPLACEVALUEENTRY']._serialized_end=644
  _globals['_FEATUREIMPUTERPARAM_MISSINGVALUERATIOENTRY']._serialized_start=646
  _globals['_FEATUREIMPUTERPARAM_MISSINGVALUERATIOENTRY']._serialized_end=702
  _globals['_FEATUREIMPUTERPARAM_MISSINGREPLACEVALUETYPEENTRY']._serialized_start=704
  _globals['_FEATUREIMPUTERPARAM_MISSINGREPLACEVALUETYPEENTRY']._serialized_end=766
  _globals['_FEATUREIMPUTERPARAM_COLSREPLACEMETHODENTRY']._serialized_start=768
  _globals['_FEATUREIMPUTERPARAM_COLSREPLACEMETHODENTRY']._serialized_end=824
  _globals['_FEATUREIMPUTATIONPARAM']._serialized_start=826
  _globals['_FEATUREIMPUTATIONPARAM']._serialized_end=950
# @@protoc_insertion_point(module_scope)
