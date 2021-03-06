
from __future__ import print_function
import tensorflow as tf
from collections import namedtuple

DTYPE_MAP = {
    u"int64": tf.int64,
    u"float32": tf.float32,
    u"string": tf.string,
    u"float64": tf.float64
}

CrossFeaInfo = namedtuple("CrossFeaInfo", ["feature_name", "feature_idx", "feature_name_idx"])


class FeatureFieldCfg(object):

    DEFAULT_TOWER_NAME = "default_tower"
    DEFAULT_FEATURE_GROUP_NAME = "default_group"

    def __init__(self, field):
        """
        :param field: dict parsed from toml
        """
        assert isinstance(field, dict)
        self._field = field

        # Do not need set var_len_field in the toml config file
        # Judge whether it is a var_len_field according to whether the pad_val is set.!!
        self._var_len_field = False

        self._num_sub_field = 1

        # you can set skipped_dims in input_layer.toml  to disable sub_field of field
        # this is very useful for experiment
        self._num_sub_field_after_skip = 1
        self._remain_dims = None
        self._tot_length_after_skip = None

        self._emb_group_name = None
        self._field_name = None
        self._pad_val = None
        self._tot_length = None
        self._mean_pooling = True

        # if ignore, the corresponding feature of this field will not appear in the model input.
        self._should_ignore = None
        self._dtype = None

        # NetInputHelper.build_model_input use this value to bucketize given raw feature value
        self._boundaries = None

        # this is useless now
        self._fea_col_type = None

        # this is useless now
        self._fea_col = None

        # parents of cross feature.
        self._parents = None  # crossed column parents

        # NetInputHelper.build_model_input use this value to decide whether hash the given raw feature value
        self._do_hash = False

        # recommendation model may have multi tower, for example: user, ad, etc.
        # this is used for this field belong to which tower.
        # if you don't need multi tower, no need to set it in input_layer.toml
        # NetInputHelper.build_model_input use this field to organise it's output dict
        self._tower_name = FeatureFieldCfg.DEFAULT_TOWER_NAME  # deprecated !!!! use feature_group instead
        self._feature_group_name = FeatureFieldCfg.DEFAULT_FEATURE_GROUP_NAME
        self._parse_field_dict()
        self._is_valid_cfg()

        # this is used for populate feature_column, for hash feature column, _emb_cfg.num_fea_values is needed.
        self._emb_cfg = None

    def set_emb_cfg(self, emb_cfg):
        self._emb_cfg = emb_cfg

    @property
    def tower_name(self):
        return self._tower_name

    @property
    def do_hash(self):
        return self._do_hash

    @property
    def parents(self):
        return self._parents

    @property
    def emb_cfg(self):
        assert self._emb_cfg is not None, "emb_cfg not set yet"
        return self._emb_cfg

    @property
    def fea_col(self):
        return self._fea_col

    @property
    def num_sub_field_after_skip(self):
        if self.num_sub_field == 1:
            return 1
        num_item = int(self.tot_length / self.num_sub_field)
        return int(len(self.remain_dims) / num_item)

    @property
    def tot_length_after_skip(self):
        return len(self.remain_dims)

    @property
    def boundaries(self):
        return self._boundaries

    @property
    def dtype(self):
        """
        :return : tensorflow dtype
        """
        return self._dtype

    @property
    def var_len_field(self):
        return self._var_len_field

    @property
    def num_sub_field(self):
        return self._num_sub_field

    @property
    def emb_group_name(self):
        return self._emb_group_name

    @property
    def field_name(self):
        return self._field_name

    @property
    def pad_val(self):
        return self._pad_val

    @property
    def tot_length(self):
        return self._tot_length

    @property
    def remain_dims(self):
        return self._remain_dims

    @property
    def should_ignore(self):
        return self._should_ignore

    @property
    def mean_pooling(self):
        return self._mean_pooling

    @property
    def feature_group_name(self):
        return self._feature_group_name

    def _parse_field_dict(self):
        self._var_len_field = FeatureFieldCfg.parse_is_var_len_field(field=self._field)
        self._num_sub_field = FeatureFieldCfg.parse_num_sub_field(field=self._field)
        self._emb_group_name = FeatureFieldCfg.parse_emb_group(self._field)
        self._field_name = FeatureFieldCfg.parse_field_name(self._field)
        self._pad_val = FeatureFieldCfg.parse_pad_val(self._field)
        self._tot_length = FeatureFieldCfg.parse_dims(self._field)
        self._mean_pooling = self._field[u'mean_pooling'] if u'mean_pooling' in self._field else True
        self._remain_dims = FeatureFieldCfg.parse_remained_dims(self._field)
        self._should_ignore = FeatureFieldCfg.parse_should_ignore(self._field)
        self._dtype = FeatureFieldCfg.parse_dtype(field=self._field)
        self._boundaries = FeatureFieldCfg.parse_boundaries(self._field)
        self._fea_col_type = FeatureFieldCfg.parse_fea_col_type(self._field)
        self._parents = FeatureFieldCfg.parse_parents_features(self._field)
        self._do_hash = FeatureFieldCfg.parse_do_hash(self._field)
        self._tower_name = FeatureFieldCfg.parse_tower_name(self._field)
        self._feature_group_name = (FeatureFieldCfg.DEFAULT_FEATURE_GROUP_NAME
                                    if u'feature_group' not in self._field else self._field[u'feature_group'])

    def _is_valid_cfg(self):
        return True

    @staticmethod
    def parse_is_var_len_field(field):
        return True if u"pad_val" in field else False

    @staticmethod
    def parse_num_sub_field(field):
        return 1 if u"num_sub_field" not in field else field[u"num_sub_field"]

    @staticmethod
    def parse_emb_group(field):
        return None if u"emb_group" not in field else field[u"emb_group"]

    @staticmethod
    def parse_field_name(field):
        return field[u"name"]

    @staticmethod
    def parse_pad_val(field):
        return None if u"pad_val" not in field else field[u"pad_val"]

    @staticmethod
    def parse_dims(field):
        return field[u"tot_length"]

    @staticmethod
    def parse_dtype(field):
        dtype_str = field[u'dtype']
        return DTYPE_MAP[dtype_str]

    @staticmethod
    def parse_remained_dims(field):
        """
        :param field:
        :return: None means no skipped dims, seems weird
        """
        field_name = FeatureFieldCfg.parse_field_name(field)
        tot_dims = FeatureFieldCfg.parse_dims(field)
        sub_fields = FeatureFieldCfg.parse_num_sub_field(field)
        all_skipped_dims = []
        if u"skipped_dims" in field:
            skipped_dims = list(map(int, field[u"skipped_dims"].split(",")))
            if sub_fields == 1:
                all_skipped_dims = skipped_dims
            else:
                for dim in skipped_dims:
                    while dim < tot_dims:
                        all_skipped_dims.append(dim)
                        dim += sub_fields
        all_skipped_dims = set(all_skipped_dims)
        tf.logging.info("field_name:{}, skipped_feature_dims:{}".format(field_name, all_skipped_dims))
        remained_dims_result = []
        for i in range(0, tot_dims):
            if i in all_skipped_dims:
                continue
            remained_dims_result.append(i)
        assert len(remained_dims_result) == (tot_dims - len(all_skipped_dims)), """
            len(remained_dims_result)={}, tot_dims - len(all_skipped_dims)={}""".format(
            len(remained_dims_result), tot_dims - len(all_skipped_dims))
        return remained_dims_result

    @staticmethod
    def parse_should_ignore(field):
        return False if u"ignore" not in field else field[u"ignore"]

    @staticmethod
    def parse_boundaries(field):
        if u'boundaries' not in field:
            return None
        boundaries = field[u'boundaries']
        if isinstance(boundaries, list):
            return boundaries
        boundaries = list(map(float, map(str.strip, boundaries.split(","))))
        return boundaries

    @staticmethod
    def parse_fea_col_type(field):
        return None if u'fea_col_type' not in field else field[u'fea_col_type']

    @staticmethod
    def parse_parents_features(field):
        if u'parents_features' not in field:
            return None
        items = map(str.strip, field[u'parents_features'].encode('utf8').split(","))
        parents = []
        for item in items:
            sub_items = item.split(":")
            fea_name = sub_items[0]
            fea_idx = None
            if len(sub_items) > 1:
                fea_idx = int(sub_items[1])
            # we build {'${fea_name}_idx_${fea_idx}': tensor} manually
            if len(sub_items) == 1:
                parents.append(CrossFeaInfo(feature_name=fea_name, feature_idx=fea_idx, feature_name_idx=fea_name))
            else:
                parents.append(CrossFeaInfo(feature_name=fea_name,
                                            feature_idx=fea_idx,
                                            feature_name_idx="{}_idx_{}".format(fea_name, fea_idx)))
        return parents

    @staticmethod
    def parse_do_hash(field):
        return False if u"do_hash" not in field else field[u"do_hash"]

    @staticmethod
    def parse_tower_name(field):
        return FeatureFieldCfg.DEFAULT_TOWER_NAME if u"tower" not in field else field[u"tower"]


class LabelFieldCfg(object):
    def __init__(self, field):
        self._field = field
        self._field_name = None
        self._fake_input_field = False
        self._dtype = None
        self._parse_field_dict()

    @property
    def tot_length(self):
        return 1

    @property
    def var_len_field(self):
        return False

    @property
    def dtype(self):
        return self._dtype

    @property
    def field_name(self):
        return self._field_name

    @property
    def fake_input_field(self):
        return self._fake_input_field

    def _parse_field_dict(self):
        self._field_name = FeatureFieldCfg.parse_field_name(field=self._field)
        self._fake_input_field = LabelFieldCfg.parse_is_fake_input_field(self._field)
        self._dtype = FeatureFieldCfg.parse_dtype(self._field)

    @staticmethod
    def parse_is_fake_input_field(field):
        return False if u"as_fake_input" not in field else field[u"as_fake_input"]


class EmbGroupCfg(object):
    def __init__(self, field):
        self._field = field
        self._name = None
        self._emb_size = None
        self._use_hash_emb_table = None
        self._num_fea_values = None
        self._use_cvm = False
        self._parse_field_dict()

    @property
    def use_cvm(self):
        return self._use_cvm

    @property
    def group_name(self):
        return self._name

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def use_hash_emb_table(self):
        return self._use_hash_emb_table

    @property
    def num_fea_values(self):
        return self._num_fea_values

    def _parse_field_dict(self):
        self._name = FeatureFieldCfg.parse_field_name(field=self._field)
        self._emb_size = EmbGroupCfg.parse_emb_size(self._field)
        self._use_hash_emb_table = EmbGroupCfg.parse_use_hash_emb_table(self._field)
        self._num_fea_values = EmbGroupCfg.parse_num_fea_values(self._field)
        self._use_cvm = EmbGroupCfg.parse_use_cvm(self._field)

    @staticmethod
    def parse_emb_size(field):
        return None if u'emb_size' not in field else field[u'emb_size']

    @staticmethod
    def parse_use_hash_emb_table(field):
        return False if u'use_hash_emb_table' not in field else field[u'use_hash_emb_table']

    @staticmethod
    def parse_num_fea_values(field):
        return None if u'num_fea_values' not in field else field[u'num_fea_values']

    @staticmethod
    def parse_use_cvm(field):
        return False if u"use_cvm" not in field else field[u"use_cvm"]
