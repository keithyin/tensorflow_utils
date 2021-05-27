# -*- coding: utf-8 -*-

from __future__ import print_function
import toml
import tensorflow as tf
from ..utils import input_layer as input_layer_utils
from tensorflow.python import math_ops
from ..net_building_blocks.cvm import ContinuousValueModel
import numpy as np
from .field_cfg import FeatureFieldCfg, LabelFieldCfg, EmbGroupCfg


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
        self._config = toml.load(config_file)
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
            if name == u"dimensions":
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
    def __init__(self, emb_config):
        """
        :param emb_config: toml config file path
        """
        self._emb_config = emb_config
        self._embeddings = {}
        self._build_embeddings()

    def _build_embeddings(self):
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
                name = group.group_name
                num_fea_values = group.num_fea_values
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
                        self._embeddings[name] = get_mutable_dense_hashtable(
                            key_dtype=tf.int64,
                            value_dtype=tf.float32,
                            shape=tf.TensorShape([emb_size]),
                            name="{}_emb_layer".format(name),
                            initializer=tf.truncated_normal_initializer(
                                0.0, 1e-2),
                            shard_num=4)
                    else:
                        self._embeddings[name] = tf.get_variable(
                            name=name, shape=[num_fea_values, emb_size],
                            dtype=tf.float32,
                            initializer=tf.initializers.random_uniform)

    def get_embeddings(self):
        assert len(self._embeddings) > 0, "call build_embedding() first"
        return self._embeddings

    def build_input_emb(self, features, feature_config, process_hooks=None):
        """
        id -> embedding, and then concat different field together.
        iterate `features`, and get corresponding config in `feature_config`
        pop field from `features` if you don't want that field present in the input embedding

        :param features: the feature part of parsed_example.
        :param feature_config: feature config
        :param process_hooks:
        """
        assert len(self._embeddings) > 0, "call build_embeddings first"
        input_tensors = {}
        # some feature may not go into this function, so we iterator features instead of config[u"feature"][u"fields"]
        # assure the order
        feature_items = sorted(list(features.items()), key=lambda x: x[0])
        feature_items = [(feature_name, ori_feature)
                         for feature_name, ori_feature in feature_items if feature_name != "dimensions"]

        tf.logging.debug("build_input_emb: feature_items: {}".format(feature_items))
        tf.logging.debug("build_input_emb: feature_config: {}".format(feature_config))

        for feature_name, ori_feature in feature_items:
            field = InputConfig.get_field_by_name(feature_config, feature_name)

            if field is None:
                tf.logging.warn(
                    """feature_name: {}, not found in self._feature_config.
                    feature items may contain label. so this warning information might be triggered.
                    """.format(
                        feature_name))
                continue
            field = FeatureFieldCfg(field)
            if field.should_ignore:
                continue
            var_len = field.var_len_field
            emb_group = field.emb_group_name

            # filter unwanted sub fields
            ori_feature = NetInputHelper.get_input_fea_of_interest(field, ori_feature)
            if field.boundaries is not None:
                ori_feature = math_ops._bucketize(ori_feature, field.boundaries,
                                                  name="bucketize_{}".format(field.field_name))
            if emb_group is not None:
                assert emb_group in self._embeddings.keys(), "emb_group: '{}' not found in embedding settings".format(
                    emb_group)

                emb_layer = self._embeddings.get(emb_group)
                if var_len:
                    tensor_val = input_layer_utils.FeaProcessor.var_len_fea_process(
                        ori_feature,
                        fea_num=field.num_sub_field_after_skip,
                        lookup_table=None,
                        emb_layer=emb_layer)

                    tensor_val = tf.reshape(
                        tensor_val, shape=[-1, np.prod(tensor_val.shape[1:])], name=field.field_name)
                else:
                    tensor_val = input_layer_utils.FeaProcessor.fix_len_fea_process(
                        ori_feature,
                        lookup_table=None,
                        emb_layer=emb_layer)
            else:

                tensor_val = ori_feature

            if process_hooks is not None and field.field_name in process_hooks:
                tensor_val = process_hooks[field.field_name](tensor_val)
            tf.logging.debug("feature_name:{}, before:{}, after:{}".format(
                feature_name,
                ori_feature,
                tensor_val))

            input_tensors[field.field_name] = tensor_val

        assert len(input_tensors) > 0, ""
        feature_items = sorted(list(input_tensors.items()), key=lambda x: x[0])
        tf.logging.info("build_input_emb, input embeddings = \n{}".format(
            "\n".join(["{} ---> {}".format(name, tensor) for name , tensor in feature_items])))
        if len(input_tensors) == 1:
            inp = tf.identity(feature_items[0][1], name="build_input_emb.input_embedding")
        else:
            inp = tf.concat([emb for _, emb in feature_items], axis=1, name="build_input_emb.input_embedding")
        tf.logging.info("build_input_emb={}".format(inp))
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

    def build_single_field_var_len_input_emb(self, features, feature_config, process_hook=None):
        """
        different from the build_input_emb.
        def build_input_emb(self):
            1. for multiple fields
            2. use mean pooling for var len field
        def build_single_field_var_len_input_emb(self):
            1. for single fields
            2. just lookup, no other process for emb
        :param features:
        :param feature_config:
        :param process_hook: signature def process_hook(emb, mask) -> (emb, mask)
        """
        assert isinstance(features, dict)
        assert len(features) == 1, "only support single field"
        field_name = list(features.keys())[0]
        field = InputConfig.get_field_by_name(feature_config, field_name)
        field = FeatureFieldCfg(field)
        assert field.var_len_field, "field:{} must be var len field".format(field_name)

        emb_group_name = field.emb_group_name
        assert emb_group_name is not None, "no emb_group for field:{}".format(field_name)

        emb_layer = self._embeddings.get(emb_group_name, None)
        assert emb_layer is not None, "{} not found in {}".format(emb_group_name, self._embeddings)

        pad_val = field.pad_val
        assert pad_val is not None, "pad_val not found in field:{}".format(field_name)
        ori_feature = NetInputHelper.get_input_fea_of_interest(field, features[field_name])
        # print("ori_feature: {}".format(ori_feature))

        emb, mask = input_layer_utils.FeaProcessor.var_len_fea_lookup(
            inp=ori_feature,
            pad_val=pad_val,
            fea_num=field.num_sub_field_after_skip,
            emb_layer=emb_layer)

        emb, mask = process_hook(emb, mask) if process_hook is not None else (emb, mask)
        return emb, mask

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
        remained_dims = field.remain_dims
        if len(remained_dims) == 0:
            return None
        if len(remained_dims) == field.tot_length:
            return ori_feature

        ori_feature = tf.gather(ori_feature, indices=tf.constant(remained_dims), axis=1)
        return ori_feature


if __name__ == '__main__':
    input_cfg = InputConfig("src/tensorflow_utils/input_layer/input_layer.toml")
    example_desc = input_cfg.build_train_example_feature_description()
    serving_input = input_cfg.build_serving_input_receiver()
    # print(example_desc)
    # print(serving_input)

    net_input_helper = NetInputHelper(input_cfg.get_emb_config())
    features = {
        "prices": tf.constant([[2.0], [1.9]], dtype=tf.float32)
    }
    # emb, mask = net_input_helper.build_single_field_var_len_input_emb(
    #     features, input_cfg.get_feature_config())
    # print(emb, mask)

    res = net_input_helper.build_input_emb(features, input_cfg.get_feature_config())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(res))
