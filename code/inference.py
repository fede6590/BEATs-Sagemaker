# Python Built-Ins:
import logging
import sys
import os
import io
import json

# External Dependencies:
import torch
import torchaudio
import torch.nn as nn
import torch.jit

# Local Dependencies:
from BEATs import BEATs, BEATsConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScriptedBEATsModel(torch.jit.ScriptModule):
    def __init__(self, cfg):
        super(ScriptedBEATsModel, self).__init__()
        self.model = BEATs(cfg)

    @torch.jit.script_method
    def forward(self, input_tensor):
        return self.model.extract_features(input_tensor, padding_mask=None)[0]


def model_fn(model_dir):
    try:
        model_name = os.path.join(model_dir, os.environ['MODEL_NAME'])
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"Model found ({model_name})")
        checkpoint = torch.load(model_name)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = ScriptedBEATsModel(cfg)
        if torch.cuda.device_count() > 1:
            gpus = torch.cuda.device_count()
            logger.info("Parallelization with {} GPUs".format(gpus))
            model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        logger.info(f"Model loaded to {device}")
        return model.to(device)
    except Exception as e:
        logger.error("Error loading model: {}".format(e))
        raise


def input_fn(request_body):
    try:
        if not request_body:
            raise ValueError("No input provided")
        logger.info("Receiving input...")
        wf, sr = torchaudio.load(io.BytesIO(request_body))
        logger.info(f'Sample rate: {sr}')
        if sr != 16000:
            wf = torchaudio.transforms.Resample(sr, 16000)(wf)
            logger.info("Resampled to 16kHz")
        logger.info("Input ready")
        return wf.to(device)
    except Exception as e:
        logger.error("Error processing input: {}".format(e))
        raise


def predict_fn(input_object, model):
    logger.info("Starting inference...")
    try:
        with torch.no_grad():
            prediction = model(input_object)
        logger.info("Inference done")
        return prediction
    except Exception as e:
        logger.error("Inference error: {}".format(e))
        raise


def output_fn(predictions, content_type):
    logger.info("Responding...")
    try:
        res = predictions.detach().cpu().numpy().tolist()
        logger.info(f"Response: {res}")
        return json.dumps(res)
    except Exception as e:
        logger.error("Error formatting output: {}".format(e))
        raise
