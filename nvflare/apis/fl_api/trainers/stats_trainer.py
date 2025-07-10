from typing import Any, Dict, Optional
import torch
from torch import nn
import pytorch_lightning as pl
from xgboost import XGBModel
from sklearn.base import BaseEstimator
from nvflare.apis.fl_api.trainers.base.fed_trainer import FedTrainer
