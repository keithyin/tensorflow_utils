# -*- coding: utf-8 -*-

from __future__ import print_function

from .. import toml
import json
import tensorflow as tf
from ..utils import input_layer as input_layer_utils
from ..utils import utils
from ..net_building_blocks.cvm import ContinuousValueModel
import numpy as np
from .field_cfg import FeatureFieldCfg, LabelFieldCfg, EmbGroupCfg
from collections import namedtuple
from tensorflow.python.ops import gen_math_ops

"""
Abstraction: 
    * feature
        * single value feature: tot_length = 1
        *          bow feature: tot_length > 1, you should never set num_sub_field on bow feature
        *     sequence feature: tot_length > 1, pad_val=SomeValue, num_sub_field >= 1
    * feature_group: feature_group is a feature group, grouping some feature together
        * can't group non-mean-pooled sequence feature with bow/singleVal feature!!!


About Config: toml or json is ok
    toml:
        [feature]
        [[feature.fields]]
        ...
        
        [label]
        [[label.fields]]
        ...
        
        [embedding]
        [[embedding.groups]]
        ...
    
    json:
        {
            "feature":{
                "fields": [{}, {}]
            },
            "label": {
                "fields": [{}, {}]
            },
            "embedding": {
                "groups": [{}, {}]
            }
        }
        
Usage:
    input_config = InputConfig("input_layer.toml")
    
    # when parse example
    def parse_function(self, record, is_training=False):
        record_desc = input_config.build_train_example_feature_description()
        record = tf.parse_single_example(record, record_desc)
        feature, label = input_config.split_parsed_record_to_feature_label(record)
        return feature, label
    
    # when build model_input. using NetInputHelper
    input_helper = NetInputHelper(input_config.get_emb_config(), shard_num=1)
    
        # features is dict of tensor that generated from dataset
        # the returned inp is also dict of tensor. one can use inp['tower_name'] 
        # to get the corresponding tower input.
        # build_model_input has default mean pooling behavior for sequence feature
        # if one want do other operation on sequence feature. 
        # one can use build_single_field_var_len_input to get 3-D tensor
    inp = input_helper.build_model_input(features, input_config.get_feature_config())
    
"""

InputTensorDesc = namedtuple('InputTensorDesc', ['tensor', 'mask'])


def is_group_features_valid(grouped_tensors):
    """

    Args:
        grouped_tensors: list of InputTensorDesc objects

    Returns:
        true if all mask is None or all mask is not None
    """
    assert isinstance(grouped_tensors[0], InputTensorDesc)
    bitmap = [t.mask is None for t in grouped_tensors]
    return sum(bitmap) == 0 or sum(bitmap) == len(bitmap)


class InputConfig(object):
    def __init__(self, config_file):
        """
        load config.toml
        used to
            1.  build_train_example_feature_description
            2. split_parsed_record_to_feature_label
            3. build_serving_input_receiver
        :param config_file: toml config file path
        """
        if config_file.endswith("toml"):
            self._config = toml.load(config_file)
        else:
            with open(config_file, mode="r") as config_file:
                self._config = json.load(config_file)
        assert u"feature" in self._config
        self._feature_config = self._config[u"feature"]
        assert u"label" in self._config
        self._label_config = self._config[u"label"]
        self._emb_config = None
        if u"embedding" in self._config:
            self._emb_config = self._config[u"embedding"]

        self._feature_desc = None
        self._label_desc = None

    def get_emb_config(self):
        return self._emb_config

    def get_feature_config(self):
        return self._feature_config

    def build_train_example_feature_description(self):
        """
        build feature description of train example. and pass the result to the tf.io.parse_single_example or
            tf.io.parse_example
        """
        self._feature_desc = self._build_feature_description(self._feature_config, False)
        self._label_desc = self._build_feature_description(self._label_config, True)
        desc = {}
        desc.update(self._feature_desc)
        desc.update(self._label_desc)
        return desc

    def split_parsed_record_to_feature_label(self, record):
        """
        :param record: the result of tf.io.parse_single_example or tf.io.parse_example
        """
        if self._feature_desc is None:
            self._feature_desc = self._build_feature_description(self._feature_config, False)
        if self._label_desc is None:
            self._label_desc = self._build_feature_description(self._label_config, True)

        features_dict = {}
        labels_dict = {}
        for name, value in record.items():
            if name in self._feature_desc:
                features_dict.update({name: value})
            elif name in self._label_desc:
                labels_dict.update({name: value})
            else:
                raise ValueError("feature_key: {} not found in feature_desc and label_desc".format(name))
        return features_dict, labels_dict

    def build_serving_input_receiver(self):
        """
        for export model. build serving_input_receiver according to config
        """
        feed_dict = {}
        assert u"fields" in self._feature_config
        assert u"fields" in self._label_config

        for field in self._feature_config[u"fields"]:
            field = FeatureFieldCfg(field)
            name = field.field_name
            if field.parents is not None:
                continue
            if name in (u"dimensions", u"dp"):
                continue
            if name in feed_dict:
                raise ValueError("duplicated feature field name: '{}'".format(name))
            feed_dict[name] = tf.placeholder(dtype=field.dtype, shape=[None, field.tot_length], name=name)
        for field in self._label_config[u"fields"]:
            field = LabelFieldCfg(field)
            if field.fake_input_field:
                name = field.field_name
                if name in feed_dict:
                    raise ValueError("duplicated feature field name: '{}'".format(name))
                feed_dict[name] = tf.placeholder(dtype=field.dtype, shape=[None, 1], name=name)
        return tf.estimator.export.ServingInputReceiver(feed_dict, feed_dict)

    @staticmethod
    def _build_feature_description(config, is_label_config):
        """
        :param config: feature_config or label_config
        """
        features_description = {}
        if u"fields" in config:
            for field in config[u"fields"]:
                if is_label_config:
                    field = LabelFieldCfg(field)
                else:
                    field = FeatureFieldCfg(field)
                    if field.parents is not None:
                        continue
                dtype = field.dtype
                tot_length = field.tot_length
                name = field.field_name
                _var_len = field.var_len_field
                if name in features_description:
                    raise ValueError("duplicated feature name '{}'".format(name))
                # metis not supported VarLenFeature now....
                # if var_len:
                #     features_description[name] = tf.io.VarLenFeature(dtype=dtype_map[dtype])
                # else:
                #     features_description[name] = tf.io.FixedLenFeature(shape=[tot_length], dtype=dtype_map[dtype])
                features_description[name] = tf.io.FixedLenFeature(shape=[tot_length], dtype=dtype)
        return features_description

    @staticmethod
    def get_field_by_name(config, name):
        assert u"fields" in config, "fields must in config, but got {}".format(config)

        for field in config[u"fields"]:
            field_name = field[u"name"]
            if field_name == name:
                return field
        return None


class NetInputHelper(object):
    def __init__(self, emb_config, is_train, shard_num=1):
        """
        attention: constructor will generate bunch of embedding matrix
        Args:
            emb_config:
            is_train: whether train mode
            shard_num:
        """
        self._emb_config = emb_config
        self._embeddings = {}
        self._build_embeddings(shard_num, is_train)

    def _build_embeddings(self, shard_num, is_train):
        """
        at the beginning the neural network, we need to build the embeddings which we will use after.
        """
        assert len(self._embeddings) == 0, "NetInputHelper object can't call this method twice"

        with tf.variable_scope(name_or_scope=None, default_name="NetInputHelperEmb"):
            if self._emb_config is None or u"groups" not in self._emb_config:
                return None
            self._embeddings = {}
            for group in self._emb_config[u"groups"]:
                group = EmbGroupCfg(group)
                names = group.group_name.strip()
                # if multiple emb group share the same config except name, we can use one EmbGroup to config it
                # the name will be like 'age,gender'
                for name in [name.strip() for name in names.split(",") if name.strip() != ""]:
                    emb_size = group.emb_size
                    if name in self._embeddings:
                        raise ValueError("duplicated embedding group name '{}'".format(name))
                    if group.use_cvm:
                        raise ValueError("has bug yet")
                        # self._embeddings[name] = ContinuousValueModel(name=name, input_dim=num_fea_values,
                        #                                               output_dim=emb_size)
                    else:
                        if group.use_hash_emb_table:
                            from tensorflow.contrib.lookup.lookup_ops import get_mutable_dense_hashtable
                            initializer = tf.truncated_normal_initializer(0.0,
                                                                          1e-2) if is_train else tf.zeros_initializer()
                            self._embeddings[name] = get_mutable_dense_hashtable(
                                key_dtype=tf.int64,
                                value_dtype=tf.float32,
                                shape=tf.TensorShape([emb_size]),
                                name="{}_emb_layer".format(name),
                                initializer=initializer,
                                shard_num=shard_num)

                        else:
                            num_fea_values = group.num_fea_values
                            self._embeddings[name] = tf.get_variable(
                                name=name, shape=[num_fea_values, emb_size],
                                dtype=tf.float32,
                                partitioner=tf.fixed_size_partitioner(shard_num, axis=0),
                                initializer=tf.random_uniform_initializer())

    def get_emb_group_cfg_by_name(self, name):
        for group in self._emb_config[u"groups"]:
            group = EmbGroupCfg(group)
            if isinstance(group.group_name, unicode):
                name_list = map(str.strip, group.group_name.encode("utf8").split(","))
            else:
                name_list = map(str.strip, group.group_name.split(","))
            if name in name_list:
                return group
        raise ValueError("emb_group_name: [{}] not found in emb_config: {}".format(name, self._emb_config))

    def get_embeddings(self):
        assert len(self._embeddings) > 0, "call build_embedding() first"
        return self._embeddings

    def build_model_input(self, features, feature_config, process_hooks=None, skip_if_not_contain=False):
        """
        id -> embedding, and then concat different field together.
        iterate `feature_config`, and get corresponding feature tensor in `features`
        difference from build_model_input: using stage1_output to do cross feature
        :param features: tensor dict. the feature part of parsed_example.
        :param feature_config: {"fields": [{...}, {...}]}
        :param process_hooks: signature func(InputTensorDesc input_tensor_desc) -> InputTensorDesc
        :param skip_if_not_contain: if the feature_config feature not in the features, Raise error or not
        :return
            dict of concatenate tower embedding. if the tower is sequence feature result[tower] got (feature, mask) pair
            for example:
            {
                "profile": tensor,
                "sku_click_activity": (tensor, mask)
            }
        """
        with tf.name_scope(name=None, default_name="build_model_input"):
            features, feature_cfg_fields = NetInputHelper.preprocess_features_and_feature_cfg(
                features, feature_config, skip_if_not_contain)

            assert len(self._embeddings) > 0, "call build_embeddings first"
            input_tensors = {}
            # assure the order

            # **************** Stage1: process boundaries, hash ****************
            stage1_process_results = {}
            for field_cfg in feature_cfg_fields:
                if field_cfg.parents is None:  # if not cross feature
                    ori_feature_tensor = features[field_cfg.field_name]
                    # ignore the skipped dims
                    feature_tensor = NetInputHelper.get_input_fea_of_interest(field_cfg, ori_feature_tensor)
                    if field_cfg.boundaries is not None:
                        feature_tensor = gen_math_ops.bucketize(feature_tensor, field_cfg.boundaries,
                                                                name="bucketize_{}".format(field_cfg.field_name))
                        feature_tensor = tf.cast(feature_tensor, dtype=tf.int64)
                    if field_cfg.do_hash:
                        feature_tensor = tf.strings.as_string(feature_tensor)
                        feature_tensor = tf.string_to_hash_bucket_strong(feature_tensor,
                                                                         num_buckets=2 ** 63 - 1, key=[2021, 2021])
                    stage1_process_results[field_cfg.field_name] = feature_tensor
            tf.logging.info("build_input_emb.stage1_process_results = ***********{}********".format(
                utils.dict_or_list_2_tuple_2_str(stage1_process_results)))
            # **************** Stage2: process cross feature ****************
            stage2_process_results = {}
            for field_cfg in feature_cfg_fields:
                if field_cfg.parents is None:
                    continue
                assert field_cfg.emb_group_name is not None, "cross features must use embedding"

                cross_feature_tensors = []
                for p in field_cfg.parents:
                    assert p.feature_name in stage1_process_results, """
                            make sure that {} has been processed at stage1
                    """.format(p.feature_name)

                    cross_feature_tensors.append(tf.strings.as_string(
                        stage1_process_results[p.feature_name][:, p.feature_idx: p.feature_idx + 1]
                        if p.feature_idx is not None else
                        stage1_process_results[p.feature_name]))

                stage2_process_results[field_cfg.field_name] = tf.string_to_hash_bucket_strong(
                    tf.string_join(cross_feature_tensors, separator="_"), key=[2021, 2021])

            tf.logging.info("build_input_emb.stage2_process_results = ***********{}********".format(
                utils.dict_or_list_2_tuple_2_str(stage2_process_results)))
            # **************** Stage3: embedding lookup ***************************
            stage3_process_input = stage1_process_results.copy()
            stage3_process_input.update(stage2_process_results)
            stage3_process_results = {}
            for field_cfg in feature_cfg_fields:
                emb_group_name = field_cfg.emb_group_name
                print("emb_group_name: {}".format(emb_group_name))
                feature_tensor = stage3_process_input[field_cfg.field_name]
                if emb_group_name is None:
                    feature_tensor = InputTensorDesc(stage3_process_input[field_cfg.field_name], mask=None)
                else:
                    assert emb_group_name in self._embeddings.keys(), """
                        emb_group: '{}' not found in embedding settings""".format(emb_group_name)

                    emb_group = self.get_emb_group_cfg_by_name(emb_group_name)

                    if emb_group.num_fea_values is not None:
                        feature_tensor = feature_tensor % emb_group.num_fea_values

                    emb_layer = self._embeddings.get(emb_group_name)
                    if field_cfg.var_len_field:
                        if field_cfg.mean_pooling:
                            feature_tensor = input_layer_utils.FeaProcessor.var_len_fea_process(
                                feature_tensor,
                                fea_num=field_cfg.num_sub_field_after_skip,
                                lookup_table=None,
                                emb_layer=emb_layer,
                                pad_val=field_cfg.pad_val)

                            feature_tensor = tf.reshape(
                                feature_tensor, shape=[-1, np.prod(feature_tensor.shape[1:])],
                                name=field_cfg.field_name)
                            feature_tensor = InputTensorDesc(tensor=feature_tensor, mask=None)
                        else:  # not do mean pooling
                            feature_tensor, mask = input_layer_utils.FeaProcessor.var_len_fea_lookup(
                                feature_tensor,
                                pad_val=field_cfg.pad_val,
                                fea_num=field_cfg.num_sub_field_after_skip,
                                lookup_table=None,
                                emb_layer=emb_layer)
                            ft_shape = tf.shape(feature_tensor)
                            feature_tensor = tf.reshape(
                                feature_tensor, shape=[ft_shape[0], ft_shape[1], np.prod(feature_tensor.shape[2:])])
                            feature_tensor = InputTensorDesc(tensor=feature_tensor, mask=mask)

                    else:  # [b, 1] -> [b, emb_size], [b, n] -> [b, n*emb_size]
                        feature_tensor = input_layer_utils.FeaProcessor.fix_len_fea_process(
                            feature_tensor,
                            lookup_table=None,
                            emb_layer=emb_layer)
                        feature_tensor = InputTensorDesc(tensor=feature_tensor, mask=None)
                stage3_process_results[field_cfg.field_name] = feature_tensor
                # **************** lookup embedding matrix DONE ***************************

            tf.logging.info("build_input_emb.stage3_process_results = ***********{}********".format(
                utils.dict_or_list_2_tuple_2_str(stage3_process_results)))
            # **************************** stage4:  post process *************************
            stage4_process_results = {}
            for field_name, feature_tensor in stage3_process_results.items():
                if process_hooks is not None and field_name in process_hooks:
                    stage4_process_results[field_name] = process_hooks[field_name](feature_tensor)
                else:
                    stage4_process_results[field_name] = feature_tensor

            assert len(stage4_process_results) > 0, ""

            tf.logging.info("build_input_emb.stage4_process_results = ***********{}********".format(
                utils.dict_or_list_2_tuple_2_str(stage4_process_results)))

            inp = NetInputHelper.organize_input_emb_with_feature_group(
                input_embs=stage4_process_results, feature_config=feature_config)
            tf.logging.info("build_input_emb.output={}".format(utils.dict_or_list_2_tuple_2_str(inp)))
            return inp

    def build_cvm_update_op(self, show_clicks):
        """
        store the cvm update_op to the tf.GraphKeys.UPDATE_OPS collection,
        do not forget to run the update ops.!!!!!
        :param show_clicks: batch sample's show_clk info [[1, 0], [1, 1], ...]
        """
        assert len(self._embeddings) > 0, "call build embeddings first"
        for k, emb in self._embeddings.items():
            if isinstance(emb, ContinuousValueModel):
                update_op = emb.update_show_clk(show_clicks)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    def build_single_field_var_len_input(self, features, feature_config, process_hook=None):
        """
        different from the build_model_input.
        def build_model_input(self):
            1. for multiple fields
            2. use mean pooling for var len field
            3. iterate feature configs
        def build_single_field_var_len_input(self):
            1. for single fields
            2. just lookup, no other process for emb
            3. find feature config according to features
        :param features: feature parsed from tf_record
        :param feature_config: parsed from toml cfg file
        :param process_hook: signature def process_hook(emb, mask) -> (emb, mask)
        """
        assert isinstance(features, dict)
        assert len(features) == 1, "only support single field"
        field_name = list(features.keys())[0]
        field_cfg = InputConfig.get_field_by_name(feature_config, field_name)
        field_cfg = FeatureFieldCfg(field_cfg)
        assert field_cfg.var_len_field, "field:{} must be var len field".format(field_name)

        emb_group_name = field_cfg.emb_group_name
        assert emb_group_name is not None, "no emb_group for field:{}".format(field_name)

        emb_layer = self._embeddings.get(emb_group_name, None)
        assert emb_layer is not None, "{} not found in {}".format(emb_group_name, self._embeddings)

        pad_val = field_cfg.pad_val
        assert pad_val is not None, "pad_val not found in field:{}".format(field_name)

        # ignore the skipped dims.
        ori_feature = NetInputHelper.get_input_fea_of_interest(field_cfg, features[field_name])
        # print("ori_feature: {}".format(ori_feature))

        emb, mask = input_layer_utils.FeaProcessor.var_len_fea_lookup(
            inp=ori_feature,
            pad_val=pad_val,
            fea_num=field_cfg.num_sub_field_after_skip,
            emb_layer=emb_layer)

        emb, mask = process_hook(emb, mask) if process_hook is not None else (emb, mask)
        return emb, mask

    @staticmethod
    def organize_input_emb_with_feature_group(input_embs, feature_config):
        """
        organize the input emb according to the tower in the feature_config
        Args:
            input_embs: tensor dict
            feature_config: feature config
        Returns:
            dict of concatenate tower embedding. if the tower is sequence feature result[tower] got (feature, mask) pair
            for example:
            {
                "profile": tensor,
                "sku_click_activity": (tensor, mask)
            }
        """
        assert isinstance(input_embs, dict)
        assert isinstance(feature_config, dict)
        tensor_group = {}
        input_embs = sorted(list(input_embs.items()), key=lambda x: x[0])
        for field_name, tensor in input_embs:
            assert isinstance(tensor, InputTensorDesc)
            field = InputConfig.get_field_by_name(feature_config, field_name)
            field = FeatureFieldCfg(field)
            if field.feature_group_name not in tensor_group:
                tensor_group[field.feature_group_name] = []
            tensor_group[field.feature_group_name].append(tensor)
        results = {}
        for feature_group_name, tensors in tensor_group.items():
            assert is_group_features_valid(tensors), """invalid grouped tensors. 
                    non-mean-pooled seq feature can't share the same feature_group with bow/singleVal feature
                    {}""".format(tensors)

            concatenated_tensors = tf.concat([t.tensor for t in tensors], axis=-1,
                                             name="input_embedding_{}".format(feature_group_name))
            if tensors[0].mask is None:
                results[feature_group_name] = concatenated_tensors
            else:
                results[feature_group_name] = (concatenated_tensors, tensors[0].mask)

        return results

    @staticmethod
    def pop_feature_from_feature_dict(feature_dict, names, keep=False):
        """
        do not modify origin feature_dict
        :param feature_dict: {"name": tensor, "name2": tensor}
        :param names: string list, feature name of interest
        :param keep: bool
        :return: (feat_of_interest_dict, feature_dict)
        """
        feature_dict = feature_dict.copy()
        assert isinstance(names, list)
        names = set(names)
        result = {}
        for name in names:
            assert name in feature_dict, "key:{} not in dict:{}".format(name, feature_dict)
            result[name] = feature_dict[name]
            if not keep:
                del feature_dict[name]
        return result, feature_dict

    @staticmethod
    def get_input_fea_of_interest(field, ori_feature):
        """
        skip the unwanted dims of ori_feature
        """
        remained_dims = field.remain_dims
        if len(remained_dims) == 0:
            return None
        if len(remained_dims) == field.tot_length:
            return ori_feature

        ori_feature = tf.gather(ori_feature, indices=tf.constant(remained_dims), axis=1)
        return ori_feature

    @staticmethod
    def preprocess_features_and_feature_cfg(features, feature_config, skip_if_not_contain=True):
        """
        preprocess_features_and_feature_cfg
        Args:
            features: dict of tensors
            feature_config: {"fields": [{...}, {...}]}
            skip_if_not_contain: boolean

        Returns:
            output_features:
            out_feature_cfg_fields:
        """
        tf.logging.warn("preprocess_features_and_feature_cfg.inp.feature_config = {}".format(
            utils.dict_or_list_2_tuple_2_str(features)))
        feature_cfg_fields = [FeatureFieldCfg(field_cfg) for field_cfg in feature_config["fields"]]
        ignored_field_names = [fea_cfg.field_name for fea_cfg in feature_cfg_fields if fea_cfg.should_ignore is True]
        tf.logging.warn("ignored field names: {}".format(ignored_field_names))
        feature_cfg_fields = [fea_cfg for fea_cfg in feature_cfg_fields if fea_cfg.should_ignore is not True]

        not_founded_feature_names = []
        output_features = {}
        out_feature_cfg_fields = []
        for fea_cfg in feature_cfg_fields:
            if fea_cfg.parents is not None:
                continue
            if fea_cfg.field_name not in features:
                if skip_if_not_contain:
                    not_founded_feature_names.append(fea_cfg.field_name)
                    continue
                else:
                    raise ValueError("feature_name:{} not found in features: {}".format(
                        fea_cfg.field_name,
                        features))
            out_feature_cfg_fields.append(fea_cfg)
            output_features[fea_cfg.field_name] = features[fea_cfg.field_name]
        tf.logging.warn("not founded field names: {}".format(not_founded_feature_names))
        tf.logging.warn("preprocess_features_and_feature_cfg.output.output_features = {}".format(
            utils.dict_or_list_2_tuple_2_str(output_features)))
        return output_features, out_feature_cfg_fields


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    input_cfg = InputConfig("src/tensorflow_utils/input_layer/input_layer_v2.toml")
    example_desc = input_cfg.build_train_example_feature_description()
    serving_input = input_cfg.build_serving_input_receiver()
    # print(example_desc)
    # print(serving_input)
    print(serving_input)

    net_input_helper = NetInputHelper(input_cfg.get_emb_config(), is_train=True)
    features = {
        "profile": tf.constant([[1, 11, 21], [2, 12, 22]], dtype=tf.int64),
        "age": tf.constant([[10], [20]], dtype=tf.int64),
        "prices": tf.constant([[2.0], [1.9]], dtype=tf.float32),
        "wm_click_sku_list": tf.constant([[1, 2, 3, 0, 0], [2, 3, 4, 5, 0]], dtype=tf.int64),
        # "wm_click_poi_list": tf.constant([[1, 2, 3, 0, 0], [2, 3, 4, 5, 0]], dtype=tf.int64),
    }
    # emb, mask = net_input_helper.build_single_field_var_len_input_emb(
    #     features, input_cfg.get_feature_config())
    # print(emb, mask)

    res = net_input_helper.build_model_input_v2(features, input_cfg.get_feature_config(), skip_if_not_contain=True)
    print(res)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(res))
        # print(sess.run(str_cat))
