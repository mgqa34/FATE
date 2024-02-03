# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: boosting-tree-model-meta.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1e\x62oosting-tree-model-meta.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"1\n\rObjectiveMeta\x12\x11\n\tobjective\x18\x01 \x01(\t\x12\r\n\x05param\x18\x02 \x03(\x01\"B\n\rCriterionMeta\x12\x18\n\x10\x63riterion_method\x18\x01 \x01(\t\x12\x17\n\x0f\x63riterion_param\x18\x02 \x03(\x01\"\xf4\x01\n\x15\x44\x65\x63isionTreeModelMeta\x12M\n\x0e\x63riterion_meta\x18\x01 \x01(\x0b\x32\x35.com.webank.ai.fate.core.mlmodel.buffer.CriterionMeta\x12\x11\n\tmax_depth\x18\x02 \x01(\x05\x12\x18\n\x10min_sample_split\x18\x03 \x01(\x05\x12\x1a\n\x12min_impurity_split\x18\x04 \x01(\x01\x12\x15\n\rmin_leaf_node\x18\x05 \x01(\x05\x12\x13\n\x0buse_missing\x18\x06 \x01(\x08\x12\x17\n\x0fzero_as_missing\x18\x07 \x01(\x08\"8\n\x0cQuantileMeta\x12\x17\n\x0fquantile_method\x18\x01 \x01(\t\x12\x0f\n\x07\x62in_num\x18\x02 \x01(\x05\"\xd5\x03\n\x15\x42oostingTreeModelMeta\x12P\n\ttree_meta\x18\x01 \x01(\x0b\x32=.com.webank.ai.fate.core.mlmodel.buffer.DecisionTreeModelMeta\x12\x15\n\rlearning_rate\x18\x02 \x01(\x01\x12\x11\n\tnum_trees\x18\x03 \x01(\x05\x12K\n\rquantile_meta\x18\x04 \x01(\x0b\x32\x34.com.webank.ai.fate.core.mlmodel.buffer.QuantileMeta\x12M\n\x0eobjective_meta\x18\x05 \x01(\x0b\x32\x35.com.webank.ai.fate.core.mlmodel.buffer.ObjectiveMeta\x12\x11\n\ttask_type\x18\x06 \x01(\t\x12\x18\n\x10n_iter_no_change\x18\x07 \x01(\x08\x12\x0b\n\x03tol\x18\x08 \x01(\x01\x12\x13\n\x0buse_missing\x18\t \x01(\x08\x12\x17\n\x0fzero_as_missing\x18\n \x01(\x08\x12\x11\n\twork_mode\x18\x0b \x01(\t\x12\x0e\n\x06module\x18\x0c \x01(\t\x12\x19\n\x11\x62oosting_strategy\x18\r \x01(\t\"w\n\x0fTransformerMeta\x12P\n\ttree_meta\x18\x01 \x01(\x0b\x32=.com.webank.ai.fate.core.mlmodel.buffer.BoostingTreeModelMeta\x12\x12\n\nmodel_name\x18\x02 \x01(\tB\x19\x42\x17\x42oostTreeModelMetaProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'boosting_tree_model_meta_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\027BoostTreeModelMetaProto'
  _globals['_OBJECTIVEMETA']._serialized_start=74
  _globals['_OBJECTIVEMETA']._serialized_end=123
  _globals['_CRITERIONMETA']._serialized_start=125
  _globals['_CRITERIONMETA']._serialized_end=191
  _globals['_DECISIONTREEMODELMETA']._serialized_start=194
  _globals['_DECISIONTREEMODELMETA']._serialized_end=438
  _globals['_QUANTILEMETA']._serialized_start=440
  _globals['_QUANTILEMETA']._serialized_end=496
  _globals['_BOOSTINGTREEMODELMETA']._serialized_start=499
  _globals['_BOOSTINGTREEMODELMETA']._serialized_end=968
  _globals['_TRANSFORMERMETA']._serialized_start=970
  _globals['_TRANSFORMERMETA']._serialized_end=1089
# @@protoc_insertion_point(module_scope)
