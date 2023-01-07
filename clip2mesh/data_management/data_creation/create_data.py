import hydra
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Union, Literal
from clip2mesh.utils import Utils, ModelsFactory, Pytorch3dRenderer, Open3dRenderer


class DataCreator:
    def __init__(self, cfg):
        # parameters from config
        self.sides: bool = cfg.sides
        self.img_tag: str = cfg.img_tag
        self.with_face: bool = cfg.with_face
        self.num_of_imgs: int = cfg.num_of_imgs
        self.output_path: Path = Path(cfg.output_path)
        self.gender: Literal["male", "female", "neutral"] = cfg.gender
        self.renderer_type: Literal["pytorch3d", "open3d"] = cfg.renderer.name

        # utils
        self.utils: Utils = Utils()
        self.models_factory: ModelsFactory = ModelsFactory(cfg.model_type)
        renderer_kwargs: Dict[str, Any] = self._get_renderer_kwargs(cfg)
        self._load_renderer(renderer_kwargs)

    def _load_renderer(self, kwargs):
        self.renderer: Union[Pytorch3dRenderer, Open3dRenderer] = self.models_factory.get_renderer(**kwargs)

    def _get_renderer_kwargs(self, cfg):
        if cfg.renderer.name == "open3d":
            # renderer_kwargs = {
            #     "verts": verts,
            #     "faces": faces,
            #     "vt": vt,
            #     "ft": ft,
            #     "paint_vertex_colors": True if cfg.model_type == "smal" else False,
            # }
            # renderer_kwargs.update(cfg.renderer.kwargs)
            # renderer = models_factory.get_renderer(**renderer_kwargs)
            raise NotImplementedError  # TODO: implement open3d renderer
        else:
            renderer_kwargs = {"py3d": True}
            renderer_kwargs.update(cfg.renderer.kwargs)

        return renderer_kwargs

    def __call__(self):
        # start creating data
        for _ in tqdm(range(self.num_of_imgs), total=self.num_of_imgs, desc="creating data"):

            # get image id
            try:
                img_id = (
                    int(
                        sorted(list(Path(self.output_path).glob("*.png")), key=lambda x: int(x.stem.split("_")[0]))[
                            -1
                        ].stem.split("_")[0]
                    )
                    + 1
                )
            except IndexError:
                img_id = 0

            # set image name
            img_name = self.img_tag if self.img_tag is not None else str(img_id)

            # get random 3DMM parameters
            model_kwargs = self.models_factory.get_random_params(with_face=self.with_face)

            # extract verts, faces, vt, ft
            verts, faces, vt, ft = self.models_factory.get_model(**model_kwargs, gender=self.gender)

            # render mesh and save image
            if self.renderer_type == "open3d":
                self.renderer.render_mesh()
                self.renderer.visualizer.capture_screen_image(f"{self.output_path}/{img_name}.png")
                self.renderer.visualizer.destroy_window()
            else:
                if self.sides:
                    for azim in [0.0, 90.0]:
                        img_suffix = "front" if azim == 0.0 else "side"
                        img = self.renderer.render_mesh(
                            verts=verts, faces=faces[None], vt=vt, ft=ft, rotate_mesh={"degrees": azim, "axis": "y"}
                        )
                        self.renderer.save_rendered_image(img, f"{self.output_path}/{img_name}_{img_suffix}.png")
                else:
                    img = self.renderer.render_mesh(verts=verts, faces=faces[None], vt=vt, ft=ft)
                    self.renderer.save_rendered_image(img, f"{self.output_path}/{img_name}.png")

            self.utils.create_metadata(metadata=model_kwargs, file_path=f"{self.output_path}/{img_name}.json")


@hydra.main(config_path="../../config", config_name="create_data")
def main(cfg):
    data_creator = DataCreator(cfg)
    data_creator()


if __name__ == "__main__":
    main()
