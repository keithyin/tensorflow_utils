import toml


class FeatureField(object):
    DEFAULT_TOWER_NAME = "default_tower"

    def __init__(self):
        self.name = None
        self.dtype = None
        self.ignore = None
        self.tot_length = None
        self.num_sub_field = None
        self.pad_val = None
        self.emb_group = None
        self.skipped_dims = None
        self.tower = FeatureField.DEFAULT_TOWER_NAME


class InputCfg(object):
    def __init__(self):
        self.feature = None


if __name__ == '__main__':
    toml.load()
