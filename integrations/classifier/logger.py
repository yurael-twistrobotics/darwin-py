import logging

# define our logger ->  https://github.com/PyTorchLightning/pytorch-lightning/issues/1503
logger = logging.getLogger(__name__)
sh = logging.StreamHandler()

sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)
logger.setLevel(logging.INFO)
