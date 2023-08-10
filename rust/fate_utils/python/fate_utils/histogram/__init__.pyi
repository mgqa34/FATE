from typing import Tuple, List, Dict, Union

class HistogramIndexer:
    def __new__(node_size: int, feature_bin_sizes: List[int]) -> "HistogramIndexer": ...
    def get_position(self, nid: int, fid: int, bid: int) -> int: ...
    def get_positions(self, nids: List[int], bid_vec: List[List[int]]) -> List[List[int]]: ...
    def get_reverse_position(self, position: int) -> Tuple[int, int, int]: ...
    def get_bin_num(self, fid: int) -> int: ...
    def get_bin_interval(self, nid: int, fid: int) -> Tuple[int, int]: ...
    def get_node_intervals(self) -> List[Tuple[int, int]]: ...
    def get_feature_position_ranges(self) -> List[Tuple[int, int]]: ...
    def splits_into_k(self, k: int) -> List[Tuple[int, Tuple[int, int], List[Tuple[int, int]]]]: ...
    def total_data_size(self) -> int: ...
    def one_node_data_size(self) -> int: ...
    def global_flatten_bin_sizes(self) -> List[int]: ...
    def flatten_in_node(self) -> "HistogramIndexer": ...
    def squeeze_bins(self) -> "HistogramIndexer": ...
    def reshape(self, feature_bin_sizes: List[int]) -> "HistogramIndexer": ...
    def get_shuffler(self, seed: int) -> "Shuffler": ...
    def get_node_size(self) -> int: ...
    def get_node_axis_stride(self) -> int: ...
    def get_feature_size(self) -> int: ...
    def get_feature_axis_stride(self) -> List[int]: ...
    def get_feature_bin_sizes(self) -> List[int]: ...
    def get_num_nodes(self) -> int: ...
    def unflatten_indexes(self) -> Dict[int, Dict[int, List[int]]]: ...

class Shuffler:
    def __new__(num_node: int, node_size: int, seed: int) -> "Shuffler": ...
    def get_global_perm_index(self) -> List[int]: ...
    def get_reverse_indexes(self, step: int, indexes: List[int]) -> List[int]: ...
    def get_shuffle_index(self, step: int, reverse: bool) -> List[int]: ...
    def reverse_index(self, index: int) -> Tuple[int, int]: ...