import hydra
from tqdm import tqdm
from pathlib import Path
from clip2mesh.utils import Utils, ModelsFactory


@hydra.main(config_path="../../config", config_name="create_data")
def main(cfg):

    assert cfg.renderer.name in ["pytorch3d", "open3d"], "Renderer not supported"

    utils = Utils()
    models_factory = ModelsFactory(cfg.model_type)

    for _ in tqdm(range(cfg.num_of_imgs), total=cfg.num_of_imgs, desc="creating data"):
        try:
            img_id = (
                int(
                    sorted(list(Path(cfg.output_path).glob("*.png")), key=lambda x: int(x.stem.split("_")[0]))[
                        -1
                    ].stem.split("_")[0]
                )
                + 1
            )
        except IndexError:
            img_id = 0
        img_name = cfg.img_tag if cfg.img_tag is not None else str(img_id)

        model_kwargs = models_factory.get_random_params(with_face=cfg.with_face)

        verts, faces, vt, ft = models_factory.get_model(**model_kwargs, gender=cfg.gender)

        if cfg.renderer.name == "open3d":
            renderer_kwargs = {
                "verts": verts,
                "faces": faces,
                "vt": vt,
                "ft": ft,
                "paint_vertex_colors": True if cfg.model_type == "smal" else False,
            }
            renderer_kwargs.update(cfg.renderer.kwargs)
            open3d_renderer = models_factory.get_renderer(**cfg.renderer.kwargs)
            open3d_renderer.render_mesh()
            open3d_renderer.visualizer.capture_screen_image(f"{cfg.output_path}/{img_name}.png")
            open3d_renderer.visualizer.destroy_window()

        else:
            if cfg.sides:
                for azim in [0.0, 90.0]:
                    img_suffix = "front" if azim == 0.0 else "side"
                    # renderer_kwargs.update(cfg.renderer.kwargs)
                    py3d_renderer = models_factory.get_renderer(py3d=True, **cfg.renderer.kwargs)
                    img = py3d_renderer.render_mesh(
                        verts=verts, faces=faces[None], vt=vt, ft=ft, rotate_mesh={"degrees": azim, "axis": "y"}
                    )
                    py3d_renderer.save_rendered_image(img, f"{cfg.output_path}/{img_name}_{img_suffix}.png")
            else:
                renderer_kwargs = {"py3d": True}
                renderer_kwargs.update(cfg.renderer.kwargs)
                py3d_renderer = models_factory.get_renderer(**renderer_kwargs)
                img = py3d_renderer.render_mesh(verts=verts, faces=faces[None], vt=vt, ft=ft)
                py3d_renderer.save_rendered_image(img, f"{cfg.output_path}/{img_name}.png")

        utils.create_metadata(metadata=model_kwargs, file_path=f"{cfg.output_path}/{img_name}.json")


if __name__ == "__main__":
    main()
