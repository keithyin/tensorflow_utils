from __future__ import print_function

from collections import OrderedDict


def fill_dict_val(mapping):
    mapping["b"] = 1
    mapping["c"] = 2
    mapping["a"] = 0
    return mapping


if __name__ == '__main__':

    for k, v in fill_dict_val(dict()).items():
        print(k, v)
    print("-----")
    for k in fill_dict_val(OrderedDict()).values():
        print(k)
