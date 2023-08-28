# Python Built-Ins:
import json
import logging
import sys
import os

# External Dependencies:
import torch
import torchaudio
import torch.nn as nn

# Local Dependencies:
from BEATs import BEATs, BEATsConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

env = os.environ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    """Load saved model from file"""
    logger.info("Loading model...")
    model_name = f"/opt/ml/model/{env['MODEL_NAME']}"
    checkpoint = torch.load(model_name)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    logger.info(f"Model loaded to {device}")
    if torch.cuda.device_count() > 1:
        logger.info("GPU count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    logger.info(f"Model loaded to {device}")
    return model.to(device)


def input_fn(request_body):
    logger.info("Receiving input...")
    wf, sr = torchaudio.load(request_body)
    if sr != 16000:
        wf = torchaudio.transforms.Resample(sr, 16000)(wf)
        logger.info("Resampled to 16kHz")
    logger.info("Input ready")
    return wf.to(device)


def predict_fn(input_object, model):
    logger.info("Starting inference...")
    with torch.no_grad():
        prediction = model.extract_features(input_object, padding_mask=None)[0]
    logger.info("Inference done")
    return prediction


def output_fn(predictions, content_type):
    """Post-process and serialize model output to API response"""
    logger.info("Responding...")
    res = predictions.detach().cpu().numpy().tolist()
    logger.info(f"Response: {res}")
    return json.dumps(res)
