from omegaconf import DictConfig
from clip2mesh.optimization.optimization import Optimization

config_file: DictConfig = "TODO"

optimization = Optimization(**config_file.optimization)
optimization.optimize()
