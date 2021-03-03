import copy
import json
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


## config class
class Config(object):
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        """Construct the config from a python dictionary"""
        config = Config()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_args(cls, args):
        """convert the namespace to dict"""
        return cls.from_dict(dict(args._get_kwargs()))

    @classmethod
    def from_json_file(cls, json_file: str):
        """Construct the config from a json file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            context = f.read()
        return cls.from_dict(json.loads(context))

    @classmethod
    def from_yaml_file(cls, yaml_file: str):
        """Construct the config from a yaml file"""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            context = f.read()
        return cls.from_dict(yaml.safe_load(context))

    def to_json_file(self, json_file):
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(self.to_json_string())

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        """convert a json object to a string"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


class Pool(nn.Module):
    """ the module to perform max, mean or sum pooling operation"""
    def __init__(
        self,
        dim,
        pool_type='max'
    ):
        super(Pool, self).__init__()
        self.dim = dim
        self.pool_type = pool_type

    def forward(self, x, keepdim=True):
        """
        Args:
            - x (Tensor): input tensor
        Returns:
            - the tensor performed pooling in the given dims
        """
        if self.pool_type == 'mean':
            xm = x.mean(self.dim, keepdim=keepdim)
        elif self.pool_type == 'max':
            xm, _ = x.max(self.dim, keepdim=keepdim)
        elif self.pool_type == 'sum':
            xm = x.sum(self.dim, keepdim=keepdim)
        else:
            raise NotImplementedError(f"pool type: {self.pool} is not supported")
        return xm


class PermEqui2(nn.Module):
    """ Permutation Equivalent Block Module """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pool: str = 'mean'
    ):
        super(PermEqui2, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        self.pool_module = Pool(dim=1, pool_type=pool)

    def forward(self, x):
        """
        Args:
            - x (Tensor): input tensor of the block
        Returns:
            Output tensor of this permutation equivalent block
        """
        xm = self.pool_module(x, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        # residual connection
        x = x - xm
        return x


class CytoSetModel(nn.Module):
    def __init__(self, args):
        """ build the model """
        super(CytoSetModel, self).__init__()
        self.args = args

        # building the model
        layers = []
        dim = args.in_dim
        for _ in range(args.nblock):
            layers.append(PermEqui2(dim, args.h_dim, args.pool))
            layers.append(nn.ELU(inplace=True))
            dim = args.h_dim
        self.enc = nn.Sequential(*layers)

        self.dec = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(args.h_dim, args.h_dim),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(args.h_dim, 1)
        )
        self.out_pool = Pool(dim=1, pool_type=args.out_pool)

    @classmethod
    def from_pretrained(cls, model_path: str, config_path: str, cache_dir=None):
        """ Construct the model from the file of a pre-trained model
        Args:
            - model_path (str): the path to the model file
            - config_path (str): the path to the configuration (args) file
            - cache_dir (bool): if cache the model file in cache dir
        """
        # model_file = file_utils.cached_path(model_path, cache_dir)
        # config_file = file_utils.cached_path(config_path, cache_dir)
        model_file, config_file = model_path, config_path
        logger.info("Loading model {} from cache at {}".format(model_path, model_file))
        # load config
        config = Config.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # instantiate model
        model = cls(config)
        state_dict = torch.load(model_file, map_location='cpu' if not torch.cuda.is_available() else None)
        model.load_state_dict(state_dict, strict=True)
        return model

    def predict(self, x):
        """ Predict the neg/pos label of samples
        Args:
            - x (Tensor): input of a tensor of shape `(batch, ncell, nmarker)`

        Returns:
            predicted label (pos/neg) of shape `(batch, )`
        """
        prob = self.forward(x)
        pred_label = torch.ge(prob, 0.5)
        return pred_label

    def forward(self, x):
        # set encoding
        x = self.enc(x)
        # pooling
        x = self.out_pool(x, keepdim=False)
        # feed forward
        x = self.dec(x).view(-1)
        x = torch.sigmoid(x)
        return x
