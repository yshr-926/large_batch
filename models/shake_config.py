# from https://github.com/hysts/pytorch_shake_shake
from collections import OrderedDict

def shake_26_2x32_config():
    model_config = OrderedDict([
            ('arch', 'shake_shake'),
            ('depth', 26),
            ('base_channels', 32),
            ('shake_forward', True),
            ('shake_backward', True),
            ('shake_image', True),
            ('input_shape', (1, 3, 32, 32)),
            # ('n_classes', 10),
        ])
    return model_config

def shake_26_2x64_config():
    model_config = OrderedDict([
            ('arch', 'shake_shake'),
            ('depth', 26),
            ('base_channels', 64),
            ('shake_forward', True),
            ('shake_backward', True),
            ('shake_image', True),
            ('input_shape', (1, 3, 32, 32)),
            # ('n_classes', 10),
        ])
    return model_config

def shake_26_2x96_config():
    model_config = OrderedDict([
            ('arch', 'shake_shake'),
            ('depth', 26),
            ('base_channels', 96),
            ('shake_forward', True),
            ('shake_backward', True),
            ('shake_image', True),
            ('input_shape', (1, 3, 32, 32)),
            # ('n_classes', 10),
        ])
    return model_config