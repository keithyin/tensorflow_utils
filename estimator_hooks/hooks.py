# coding=utf-8
import tensorflow as tf
from tensorflow.python.training import session_run_hook
from collections import Counter
import numpy as np


def print_distribution_dict(dist_dict):
    o_str = "{"
    key_vals = list(dist_dict.items())
    key_vals = sorted(key_vals, key=lambda x: x[1], reverse=True)
    for k, v in key_vals:
        item_str = "{}: {:.2f}%, ".format(k, v * 100)
        o_str += item_str
    o_str += "}"
    return o_str


class LabelDistHook(session_run_hook.SessionRunHook):
    def __init__(self, name, label_tensor, mask_tensor, log_step=1e8, reset_step=None,
                 message_pusher=None):
        assert len(label_tensor.shape) == len(mask_tensor.shape), "label_tensor.shape={}, mask.shape={}".format(
            label_tensor.shape, mask_tensor.shape)

        self._global_step = tf.train.get_global_step()
        assert self._global_step is not None, "no global step, no happy"

        self._label_tensor = label_tensor
        self._mask_tensor = mask_tensor
        self._reset_step = reset_step
        self._counter = Counter()
        self._log_step = log_step
        self._name = name
        self._last_global_step = 0
        self._inner_step = 0
        self._message_pusher = message_pusher

    def _reset(self):
        tf.logging.info("LabelDistHook reset counter")
        self._inner_step = 0
        self._counter.clear()

    def begin(self):
        self._reset()

    def before_run(self, run_context):
        if self._reset_step is not None and (self._inner_step + 1) % self._reset_step == 0:
            self._log(force_print=True)
            self._reset()
        self._inner_step += 1
        return session_run_hook.SessionRunArgs(fetches=[self._label_tensor, self._mask_tensor, self._global_step])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        # assert isinstance(run_values, session_run_hook.SessionRunValues)
        label, mask, global_step = run_values.results
        self._last_global_step = global_step
        # assert isinstance(label, np.ndarray), "type of label is {}, {}".format(type(label), label)
        # assert isinstance(mask, np.ndarray), "type of mask is {}, {}".format(type(mask), mask)
        label = label.flatten().astype(dtype=np.int)
        mask = mask.flatten().astype(dtype=np.int)
        assert len(label) == len(mask), "len(label)={}, len(mask)={}. not equal".format(len(label), len(mask))
        label = label[mask == 1]
        self._counter.update(label.tolist())
        self._log()

    def end(self, session=None):
        tf.logging.info("SESSION RUN END!")
        self._log(force_print=True)

    def _log(self, force_print=False):
        if self._inner_step % self._log_step == 0 or force_print:
            tot_count = float(sum(self._counter.values()))
            label_dist = {}
            if tot_count > 0:
                label_dist = dict([(key, val / tot_count) for key, val in self._counter.items()])
            msg = "Distribution: {}, global_step: {}, inner_step: {}, tot_ins: {}, label_dist: {}".format(
                self._name, self._last_global_step, self._inner_step,
                int(tot_count), print_distribution_dict(label_dist))

            tf.logging.info(msg)
            if self._message_pusher is not None:
                self._message_pusher.push_text(msg)


class GroupAucHook(session_run_hook.SessionRunHook):
    def __init__(self, name, group_tensor, label_tensor, pred_tensor,
                 log_step=1e8, num_buckets=10240, reset_step=None, message_pusher=None):

        tf.logging.info("GroupAucHook, name: {}, group_tensor: {}, label_tensor: {}, pred_tensor: {}".format(
            name, group_tensor, label_tensor, pred_tensor
        ))
        assert len(group_tensor.shape) == len(label_tensor.shape)
        assert len(label_tensor.shape) == len(pred_tensor.shape)
        self._global_step = tf.train.get_global_step()
        assert self._global_step is not None, "no global step, no happy"

        self._group_tensor = group_tensor
        self._label_tensor = label_tensor
        self._pred_tensor = pred_tensor
        self._log_step = log_step
        self._reset_step = reset_step
        self._inner_step = 0
        self._last_global_step = 0
        self._num_buckets = num_buckets
        self._group_aucs = {}
        self._name = name
        self._message_pusher = message_pusher

    def _reset(self):
        tf.logging.info("GroupAuc reset")
        self._inner_step = 0
        self._group_aucs = {}

    def begin(self):
        self._reset()

    def before_run(self, run_context):
        if self._reset_step is not None and (self._inner_step + 1) % self._reset_step == 0:
            self._log(force_print=True)
            self._reset()
        self._inner_step += 1
        return session_run_hook.SessionRunArgs(
            fetches=[self._group_tensor, self._label_tensor, self._pred_tensor, self._global_step])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        group, label, pred, global_step = run_values.results
        self._last_global_step = global_step
        group = group.flatten()
        label = label.flatten()
        pred = pred.flatten()
        distinct_group = set(group.tolist())
        for g in distinct_group:
            if g not in self._group_aucs:
                self._group_aucs[g] = Auc(self._num_buckets)
        for g in distinct_group:
            selector = group == g
            self._group_aucs[g].Update(labels=label[selector], predicts=pred[selector])

        self._log()

    def end(self, session):
        tf.logging.info("SESSION RUN END!")
        self._log(force_print=True)

    def _log(self, force_print=False):
        if self._inner_step % self._log_step == 0 or force_print:
            tot_ins = 0
            for group_name, auc in self._group_aucs.items():
                tot_ins += auc.GetNumIns()
            tot_ins = float(tot_ins)

            group_auc = 0
            detailed_auc_infos = []

            for group_name, auc in self._group_aucs.items():
                proportional = auc.GetNumIns() / tot_ins
                this_group_auc = auc.Compute()
                group_auc += proportional * this_group_auc

                detailed_auc_infos.append([group_name,
                                           auc.GetNumIns(), proportional * 100, this_group_auc])

            info = "GroupAucInfo: {}, global_step: {}, inner_step:{}, tot_ins: {}, GROUP_AUC: {:.4f}\n".format(
                self._name,
                self._last_global_step,
                self._inner_step,
                int(tot_ins),
                group_auc)

            info_fmt = "group: {}, group_ins: {}, pct: {:.4f}%, auc: {:.4f}\n"
            detailed_auc_infos = sorted(detailed_auc_infos, key=lambda x: x[1], reverse=True)

            for item in detailed_auc_infos:
                info += (info_fmt.format(item[0], item[1], item[2], item[3]))
            tf.logging.info(info)

            if self._message_pusher is not None:
                self._message_pusher.push_text(info)


class Auc(object):
    def __init__(self, num_buckets):
        self._num_buckets = num_buckets
        self._table = np.zeros(shape=[2, self._num_buckets])
        self._num_ins = 0

    def Reset(self):
        self._num_ins = 0
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Update(self, labels, predicts):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.astype(np.int)
        num_0 = sum(labels == 0)
        num_1 = sum(labels == 1)
        tot_label = np.prod(labels.shape)
        assert (num_0 + num_1) == tot_label, \
            "label must be 0 or 1, but got: {}. num_0: {}, num_1: {}, tot_label: {}".format(
                labels,
                num_0,
                num_1,
                tot_label)
        assert sum(predicts > 1) == 0 and sum(predicts < 0) == 0, \
            "predicts must in [0, 1], but got: {}".format(predicts)

        predicts = self._num_buckets * predicts
        buckets = np.round(predicts).astype(np.int)
        buckets = np.where(buckets < self._num_buckets,
                           buckets, self._num_buckets - 1)

        for i in range(len(labels)):
            self._table[labels[i], buckets[i]] += 1

        self._num_ins += len(labels)

    def Compute(self):
        tn = 0
        tp = 0
        area = 0
        for i in range(self._num_buckets):
            new_tn = tn + self._table[0, i]
            new_tp = tp + self._table[1, i]
            # self._table[1, i] * tn + self._table[1, i]*self._table[0, i] / 2
            area += (new_tp - tp) * (tn + new_tn) / 2
            tn = new_tn
            tp = new_tp
        if tp < 1e-3 or tn < 1e-3:
            return -0.5  # 样本全正例，或全负例
        return area / (tn * tp)

    def GetNumIns(self):
        return self._num_ins


class RegressionGroupInfo(object):
    def __init__(self, num_ins=0, tot_bias=0, tot_mae=0, tot_mse=0):
        self.num_ins = num_ins
        self.tot_bias = tot_bias
        self.tot_mae = tot_mae
        self.tot_mse = tot_mse


class RegressionHook(session_run_hook.SessionRunHook):
    def __init__(self, name, group_tensor, label_tensor, pred_tensor, mask_tensor, log_step=1e8, reset_step=None,
                 message_pusher=None):
        assert len(label_tensor.shape) == len(mask_tensor.shape), "label_tensor.shape={}, mask.shape={}".format(
            label_tensor.shape, mask_tensor.shape)

        self._global_step = tf.train.get_global_step()
        assert self._global_step is not None, "no global step, no happy"

        self._group_tensor = group_tensor
        self._label_tensor = label_tensor
        self._pred_tensor = pred_tensor
        self._mask_tensor = mask_tensor
        self._reset_step = reset_step
        self._log_step = log_step
        self._name = name
        self._message_pusher = message_pusher

        self._last_global_step = 0
        self._inner_step = 0
        self._group_infos = {}

    def _reset(self):
        tf.logging.info("RegressionMetrics reset")
        self._inner_step = 0
        self._group_infos.clear()

    def begin(self):
        self._reset()
        pass

    def before_run(self, run_context):
        if self._reset_step is not None and (self._inner_step + 1) % self._reset_step == 0:
            self._log(force_print=True)
            self._reset()
        self._inner_step += 1
        return session_run_hook.SessionRunArgs(
            fetches=[self._group_tensor, self._label_tensor, self._pred_tensor,
                     self._mask_tensor, self._global_step])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        group, label, pred, mask, global_step = run_values.results
        self._last_global_step = global_step
        mask = mask.flatten()
        selector = mask == 1
        group = group.flatten()[selector]
        label = label.flatten()[selector]
        pred = pred.flatten()[selector]
        distinct_group = set(group.tolist())
        for g in distinct_group:
            if g not in self._group_infos:
                self._group_infos[g] = RegressionGroupInfo(num_ins=0, tot_bias=0, tot_mae=0, tot_mse=0)
        for g in distinct_group:
            group_ins_selector = group == g
            group_label = label[group_ins_selector]
            group_pred = pred[group_ins_selector]
            point_wise_bias = group_label - group_pred
            self._group_infos[g].num_ins += len(group_label)
            self._group_infos[g].tot_bias += np.sum(point_wise_bias)
            self._group_infos[g].tot_mae += np.sum(np.abs(point_wise_bias))
            self._group_infos[g].tot_mse += np.sum(np.square(point_wise_bias))

        self._log()

    def end(self, session):
        self._log(force_print=True)
        pass

    def _log(self, force_print=False):
        if self._inner_step % self._log_step == 0 or force_print:
            tot_ins = 0
            tot_bias = 0
            tot_mae = 0
            tot_mse = 0
            for group_info in self._group_infos.values():
                tot_ins += group_info.num_ins
                tot_bias += group_info.tot_bias
                tot_mae += group_info.tot_mae
                tot_mse += group_info.tot_mse
            if tot_ins == 0:
                return

            tot_ins = float(tot_ins)
            info = "RegressionMetrics: {}, global_step: {}, inner_step: {}, tot_ins: {}, " \
                   "mean_bias: {:.3f}, mean_mae: {:.3f}, mean_mse: {:.3f}\n".format(
                    self._name,
                    self._last_global_step,
                    self._inner_step,
                    int(tot_ins),
                    tot_bias / tot_ins,
                    tot_mae / tot_ins,
                    tot_mse / tot_ins)
            group_infos = sorted(list(self._group_infos.items()), key=lambda x: x[1].num_ins, reverse=True)
            group_info_fmt = "group: {}, num_ins: {}, pct: {:.4f}%, " \
                             "mean_bias: {:.3f}, mean_mae: {:.3f}, mean_mse: {:.3f}\n"
            for group_info in group_infos:
                group_name = group_info[0]
                num_ins = group_info[1].num_ins
                if num_ins < 1:
                    continue
                num_ins = float(num_ins)
                tot_bias = group_info[1].tot_bias
                tot_mae = group_info[1].tot_mae
                tot_mse = group_info[1].tot_mse
                group_msg = group_info_fmt.format(
                    group_name, int(num_ins), num_ins / tot_ins * 100,
                    tot_bias / num_ins, tot_mae / num_ins, tot_mse / num_ins)
                info += group_msg

            tf.logging.info(info)
            if self._message_pusher is not None:
                self._message_pusher.push_text(info)
