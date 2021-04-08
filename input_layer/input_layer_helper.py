# -*- coding: utf-8 -*-

from __future__ import print_function
import toml
import tensorflow as tf
from tensorflow.keras import layers
from ..utils import input_layer as input_layer_utils
from ..net_building_blocks.cvm import ContinuousValueModel
import numpy as np

dtype_map = {
    u"int64": tf.int64,
    u"float32": tf.float32,
    u"string": tf.string,
    u"float64": tf.float64
}


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
        self._feature_desc = self._build_feature_description(self._feature_config)
        self._label_desc = self._build_feature_description(self._label_config)
        desc = {}
        desc.update(self._feature_desc)
        desc.update(self._label_desc)
        return desc

    def split_parsed_record_to_feature_label(self, record):
        """
        :param record: the result of tf.io.parse_single_example or tf.io.parse_example
        """
        if self._feature_desc is None:
            self._feature_desc = self._build_feature_description(self._feature_config)
        if self._label_desc is None:
            self._label_desc = self._build_feature_description(self._label_config)

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
            name = field[u"name"]
            tot_length = 1 if u"tot_length" not in field else field[u"tot_length"]
            dtype = field[u"dtype"]
            if name in feed_dict:
                raise ValueError("duplicated feature field name: '{}'".format(name))
            feed_dict[name] = tf.placeholder(dtype=dtype_map[dtype], shape=[None, tot_length], name=name)
        for field in self._label_config[u"fields"]:
            if u"as_fake_input" in field and field[u"as_fake_input"] == 1:
                name = field[u"name"]
                dtype = field[u"dtype"]
                if name in feed_dict:
                    raise ValueError("duplicated feature field name: '{}'".format(name))
                feed_dict[name] = tf.placeholder(dtype=dtype_map[dtype], shape=[None, 1], name=name)
        return tf.estimator.export.ServingInputReceiver(feed_dict, feed_dict)

    @staticmethod
    def _build_feature_description(config):
        """
        :param config: feature_config or label_config
        """
        features_description = {}
        if u"fields" in config:
            for field in config[u"fields"]:
                dtype = field[u"dtype"]
                tot_length = 1 if u"tot_length" not in field else field[u"tot_length"]
                name = field[u"name"]
                _var_len = InputConfig.is_var_len_field(field)
                if name in features_description:
                    raise ValueError("duplicated feature name '{}'".format(name))
                # metis not supported VarLenFeature now....
                # if var_len:
                #     features_description[name] = tf.io.VarLenFeature(dtype=dtype_map[dtype])
                # else:
                #     features_description[name] = tf.io.FixedLenFeature(shape=[tot_length], dtype=dtype_map[dtype])
                features_description[name] = tf.io.FixedLenFeature(shape=[tot_length], dtype=dtype_map[dtype])
        return features_description

    @staticmethod
    def get_field_by_name(config, name):
        assert u"fields" in config, "fields must in config, but got {}".format(config)

        for field in config[u"fields"]:
            field_name = field[u"name"]
            if field_name == name:
                return field
        return None

    @staticmethod
    def is_fake_input_field(field):
        if u"as_fake_input" in field and field[u"as_fake_input"] is True:
            return True
        return False

    @staticmethod
    def is_var_len_field(field):
        return True if u"pad_val" in field else False

    @staticmethod
    def num_sub_field(field):
        return 1 if u"num_sub_field" not in field else field[u"num_sub_field"]

    @staticmethod
    def emb_group(field):
        return None if u"emb_group" not in field else field[u"emb_group"]

    @staticmethod
    def field_name(field):
        return field[u"name"]


class NetInputHelper(object):
    def __init__(self, emb_config):
        """
        :param emb_config: toml config file path
        """
        self._emb_config = emb_config
        self._embeddings = None

    def build_embeddings(self):
        """
        at the beginning the neural network, we need to build the embeddings which we will use after.
        """
        assert self._embeddings is None, "NetInputHelper object can't call this method twice"

        with tf.variable_scope(name_or_scope=None, default_name="NetInputHelperEmb"):
            if self._emb_config is None or u"groups" not in self._emb_config:
                return None
            self._embeddings = {}
            for group in self._emb_config[u"groups"]:
                name = group[u"name"]
                num_fea_values = group[u"num_fea_values"]
                emb_size = group[u"emb_size"]
                if name in self._embeddings:
                    raise ValueError("duplicated embedding group name '{}'".format(name))
                if u"use_cvm" in group and group[u"use_cvm"]:
                    self._embeddings[name] = ContinuousValueModel(name=name, input_dim=num_fea_values,
                                                                  output_dim=emb_size)
                else:
                    self._embeddings[name] = layers.Embedding(input_dim=num_fea_values, output_dim=emb_size)

    def get_embeddings(self):
        assert self._embeddings is not None, "call build_embedding() first"
        return self._embeddings

    def build_input_emb(self, features, feature_config, process_hooks=None):
        """
        id -> embedding, and then concat different field together.
        :param features: the feature part of parsed_example.
        :param feature_config: feature config
        :param process_hooks:
        """
        assert self._embeddings is not None, "call build_embeddings first"
        input_tensors = []
        # some feature may not go into this function, so we iterator features instead of config[u"feature"][u"fields"]
        # assure the order
        feature_items = sorted(list(features.items()), key=lambda x: x[0])
        feature_items = [(feature_name, ori_feature)
                         for feature_name, ori_feature in feature_items if feature_name != "dimensions"]

        tf.logging.info("build_input_emb: feature_items: {}".format(feature_items))
        tf.logging.info("build_input_emb: feature_config: {}".format(feature_config))

        for feature_name, ori_feature in feature_items:
            field = InputConfig.get_field_by_name(feature_config, feature_name)
            if field is None:
                tf.logging.warn("feature_name: {}, not found in self._feature_config, make sure it is expected")
                continue
            name = InputConfig.field_name(field)
            var_len = InputConfig.is_var_len_field(field)
            num_sub_field = InputConfig.num_sub_field(field)
            emb_group = InputConfig.emb_group(field)
            if emb_group is not None:
                assert emb_group in self._embeddings.keys(), "emb_group: '{}' not found in embedding settings".format(
                    emb_group)

                emb_layer = self._embeddings.get(emb_group)
                if var_len:
                    tensor_val = input_layer_utils.FeaProcessor.var_len_fea_process(
                        ori_feature,
                        fea_num=num_sub_field,
                        lookup_table=None,
                        emb_layer=emb_layer)
                    tensor_val = tf.reshape(tensor_val, shape=[-1, np.prod(tensor_val.shape[1:])])
                else:
                    tensor_val = input_layer_utils.FeaProcessor.fix_len_fea_process(
                        ori_feature,
                        lookup_table=None,
                        emb_layer=emb_layer)
            else:
                continue

            if process_hooks is not None and name in process_hooks:
                tensor_val = process_hooks[name](tensor_val)
            tf.logging.info("feature_name:{}, before:{}, after:{}".format(
                feature_name,
                ori_feature,
                tensor_val))

            input_tensors.append(tensor_val)

        assert len(input_tensors) > 0, ""
        tf.logging.info("input_embeddings: {}".format(input_tensors))
        if len(input_tensors) == 1:
            inp = input_tensors[0]
        else:
            inp = tf.concat(input_tensors, axis=1)
        tf.logging.info("model input: {}".format(inp))
        return inp

    def build_cvm_update_op(self, show_clicks):
        """
        store the cvm update_op to the tf.GraphKeys.UPDATE_OPS collection,
        do not forget to run the update ops.!!!!!
        :param show_clicks: batch sample's show_clk info [[1, 0], [1, 1], ...]
        """
        assert self._embeddings is not None, "call build embeddings first"
        for k, emb in self._embeddings.items():
            if isinstance(emb, ContinuousValueModel):
                update_op = emb.update_show_clk(show_clicks)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)


if __name__ == '__main__':
    input_cfg = InputConfig("src/input_layer/input_layer.toml")
    example_desc = input_cfg.build_train_example_feature_description()
    serving_input = input_cfg.build_serving_input_receiver()
    print(example_desc)
    print(serving_input)