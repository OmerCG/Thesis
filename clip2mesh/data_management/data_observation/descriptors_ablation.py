import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig
from typing import List, Literal
from hydra import compose, initialize
from clip2mesh.optimizations.train_mapper import train
from clip2mesh.data_management.data_observation.evaluate_performance import EvaluatePerformance
from clip2mesh.data_management.data_observation.choosing_descriptors_arik import ChoosingDescriptorsArik


class DescriptorsAblation:
    def __init__(
        self,
        descriptors_options: List[int],
        model_type: Literal["smplx", "flame", "smal"],
        data_path: str,
        output_path: str,
        descriptors_clusters_json: str,
        batch_size: int,
        renderer_kwargs: DictConfig,
        optimize_feature: Literal["betas", "flame_expression", "flame_shape", "beta"],
        min_slider_value: int = 15,
        max_slider_value: int = 30,
        effect_threshold: float = 0.5,
        models_dir: str = "/home/nadav2/dev/repos/Thesis/pre_production",
    ):
        self.descriptors_options = descriptors_options
        self.model_type = model_type
        self.batch_size = batch_size
        self.optimize_feature = optimize_feature
        self.min_slider_value = min_slider_value
        self.max_slider_value = max_slider_value
        self.effect_threshold = effect_threshold
        self.renderer_kwargs = renderer_kwargs
        self.models_dir: Path = Path(models_dir)
        self.output_path: Path = Path(output_path)
        self.data_path: Path = Path(data_path)
        self.descriptors_clusters_json: Path = Path(descriptors_clusters_json)

        self._get_logger()

    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    def get_descriptors(self, num_of_descriptors: int):
        choosing_descriptors = ChoosingDescriptorsArik(
            images_dir=self.data_path,
            max_num_of_descriptors=num_of_descriptors,
            min_num_of_descriptors=num_of_descriptors,
            descriptors_clusters_json=self.descriptors_clusters_json,
        )
        descriptors = choosing_descriptors.choose()
        return descriptors

    def train_mapper(self, descriptors: List[str]):
        with initialize(config_path="../../config"):
            config = compose(config_name="train")
            config.tensorboard_logger.name = f"{self.model_type}_{len(descriptors)}_descriptors"
            config.dataloader.batch_size = self.batch_size
            config.dataset.data_dir = self.data_path
            config.dataset.optimize_feature = self.optimize_feature
            config.dataset.labels_to_get = descriptors
            config.model_conf.num_stats = len(descriptors)
            train(config)
        return f"{self.model_type}_{len(descriptors)}_descriptors"

    def evaluate_performance(self, run_name: str):
        evaluator = EvaluatePerformance(
            model_type=self.model_type,
            min_value=self.min_slider_value,
            max_value=self.max_slider_value,
            effect_threshold=self.effect_threshold,
            renderer_kwargs=self.renderer_kwargs,
            model_path=self.models_dir / run_name,
            out_path=self.output_path,
            optimize_feature=self.optimize_feature,
        )
        evaluator.evaluate()

    def __call__(self):

        for num_of_descriptors in self.descriptors_options:
            self.logger.info(f"Starting with {num_of_descriptors} descriptors")
            descriptors = self.get_descriptors(num_of_descriptors)
            self.logger.info(f"Chosen descriptors: {descriptors}")
            self.logger.info(f"Starting training")
            run_name = self.train_mapper(descriptors)
            self.logger.info(f"Starting evaluation")
            self.evaluate_performance(run_name)
            self.logger.info(f"Finished with {num_of_descriptors} descriptors")
            print()


@hydra.main(config_path="../../config", config_name="descriptors_ablation")
def main(cfg: DictConfig):
    descriptors_ablation = DescriptorsAblation(**cfg)
    descriptors_ablation()
