import json
import torch
import tkinter
import hydra
import numpy as np
from pathlib import Path
from PIL import ImageTk, Image
from pytorch3d.io import save_obj
from typing import Dict, Any
from clip2mesh.utils import Utils, ModelsFactory


class SlidersApp:
    def __init__(self, cfg):

        self.root: str = None
        self.img_label: str = None
        self.device: str = cfg.device
        self.texture: str = cfg.texture
        # self.num_coeffs: int = cfg.num_coef

        self.on_parameters: bool = cfg.on_parameters

        assert cfg.model_type in ["smplx", "flame", "smal"], "Model type should be smplx, flame or smal"
        self.model_type = cfg.model_type

        self.outpath = None
        if cfg.out_dir is not None:
            if not Path(cfg.out_dir).exists():
                Path(cfg.out_dir).mkdir(parents=True)
            try:
                img_id = int(sorted(list(Path(cfg.out_dir).glob("*.png")), key=lambda x: int(x.stem))[-1].stem) + 1
            except IndexError:
                img_id = 0
            self.outpath = Path(cfg.out_dir) / f"{img_id}.png"

        self.params = []
        self.utils = Utils()
        self.models_factory = ModelsFactory(self.model_type)
        self.gender = cfg.gender
        self.with_face = cfg.with_face
        self.model_kwargs = self.models_factory.get_default_params(cfg.with_face)
        self.verts, self.faces, self.vt, self.ft = self.models_factory.get_model(**self.model_kwargs)
        self.renderer_kwargs = {"py3d": True}
        self.renderer_kwargs.update(cfg.renderer_kwargs)
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
        img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
        self.img = self.adjust_rendered_img(img)

        self.default_dist = cfg.renderer_kwargs["dist"]
        self.default_azim = cfg.renderer_kwargs["azim"]
        self.default_elev = cfg.renderer_kwargs["elev"]

        self.production_scales = []
        self.camera_scales = {}
        self.initialize_params()

        self.ignore_random_jaw = True if cfg.model_type == "flame" and cfg.with_face else False

        if cfg.model_path is not None:
            self.model, labels = self.utils.get_model_to_eval(cfg.model_path)
            self.mean_values = {label[0]: 20 for label in labels}
            self.input_for_model = torch.tensor(list(self.mean_values.values()), dtype=torch.float32)[None]

    def initialize_params(self):
        if self.on_parameters:
            if self.model_type == "smplx":
                self.betas = self.model_kwargs["betas"]
                self.expression = self.model_kwargs["expression"]
                self.params.append(self.betas)
                self.params.append(self.expression)

            elif self.model_type == "flame":
                if self.with_face:
                    self.face_expression = self.model_kwargs["expression_params"][..., :10]
                    self.params.append(self.face_expression)
                else:
                    self.face_shape = self.model_kwargs["shape_params"][..., :10]
                    self.params.append(self.face_shape)

            else:
                self.beta = self.model_kwargs["beta"]
                self.params.append(self.beta)

    def update_betas(self, idx: int):
        def update_betas_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.betas[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_smplx_model(
                betas=self.betas, expression=self.expression, gender=self.gender
            )
            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_betas_values

    def update_face_shape(self, idx: int):
        def update_face_shape_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.face_shape[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(shape_params=self.face_shape)

            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_face_shape_values

    def update_face_expression(self, idx: int):
        def update_face_expression_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.face_expression[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(
                expression_params=self.face_expression[..., :10]
            )
            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_face_expression_values

    def update_beta_shape(self, idx: int):
        def update_beta_shape_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.beta[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_smal_model(beta=self.beta)

            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_beta_shape_values

    def update_labels(self, idx: int):
        def update_labels_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.input_for_model[0, idx] = value
            # print(self.input_for_model)  # for debug
            with torch.no_grad():
                out = self.model(self.input_for_model.to(self.device))
                if self.model_type == "smplx":
                    betas = out.cpu()
                    expression = torch.zeros(1, 10)
                    body_pose = torch.eye(3).expand(1, 21, 3, 3)
                    self.verts, self.faces, self.vt, self.ft = self.utils.get_smplx_model(
                        betas=betas, body_pose=body_pose, expression=expression, gender=self.gender
                    )
                elif self.model_type == "flame":
                    if self.with_face:
                        self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(
                            expression_params=out.cpu()
                        )
                    else:
                        self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(shape_params=out.cpu())

                else:
                    self.verts, self.faces, self.vt, self.ft = self.utils.get_smal_model(beta=out.cpu())

            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_labels_values

    def adjust_rendered_img(self, img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def add_texture(self):
        if self.texture is not None:
            self.renderer_kwargs.update({"tex_path": self.texture})
            self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

    def remove_texture(self):
        self.renderer_kwargs.update({"tex_path": None})
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
        img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def update_camera_zoom(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer_kwargs.update({"dist": value})
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
        img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def update_camera_azim(self, value: float):
        if isinstance(value, str):
            value = float(value)
        rotate_mesh_kwargs = {"degrees": value, "axis": "y"}
        img = self.renderer.render_mesh(
            verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft, rotate_mesh=rotate_mesh_kwargs
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def update_camera_elev(self, value: float):
        if isinstance(value, str):
            value = float(value)
        rotate_mesh_kwargs = {"degrees": value, "axis": "x"}
        img = self.renderer.render_mesh(
            verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft, rotate_mesh=rotate_mesh_kwargs
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def get_key_for_model(self) -> str:
        return self.models_factory.get_key_name_for_model(self.model_type)

    def mouse_wheel(self, action):
        """
        control the scrollbar using the mousewheel
        """
        scroll = -1 if action.delta > 0 else 1
        self.parameters_canvas.yview(scroll, "units")

    def save_png(self):
        if self.on_parameters:  # TODO
            self.renderer.visualizer.capture_screen_image(self.outpath.as_posix())
            key = self.get_key_for_model()
            concat_params = self._zeros_to_concat()
            params = {key: [self.params[0].tolist()[0] + concat_params]}
            with open(self.outpath.with_suffix(".json"), "w") as f:
                json.dump(params, f)
        else:
            self.img = np.array(self.img)
            self.renderer.save_rendered_image(self.img, self.outpath.as_posix())
        new_img_id = int(self.outpath.stem) + 1
        self.outpath = self.outpath.parent / f"{new_img_id}.png"

    def save_obj(self):
        if self.outpath is not None:
            if self.outpath.suffix == ".obj":
                obj_path = self.outpath
            elif self.outpath.suffix == ".png":
                obj_path = self.outpath.with_suffix(".obj")
        else:
            obj_path = "./out.obj"
        obj_path = str(obj_path)
        save_obj(
            f=obj_path,
            verts=torch.tensor(self.verts).squeeze(),
            faces=torch.tensor(self.faces).squeeze(),
        )
        if self.outpath is not None:
            new_img_id = int(self.outpath.stem) + 1
            self.outpath = self.outpath.parent / f"{new_img_id}.png"

    def random_button(self):
        if self.on_parameters:
            random_params = self.models_factory.get_random_params(self.with_face)[self.get_key_for_model()][0, :10]
            scales_list = self.production_scales if not self.ignore_random_jaw else self.production_scales[:-1]
            for idx, scale in enumerate(scales_list):
                scale.set(random_params[idx].item())

    def update_expression(self, idx: int):
        def update_expression_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.expression[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_smplx_model(
                betas=self.betas, body_pose=self.body_pose, expression=self.expression
            )
            self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)

        return update_expression_values

    def reset_parameters(self):
        if self.on_parameters:
            for scale in self.production_scales:
                scale.set(0)
        else:
            for label in self.production_scales:
                label.set(20)

    def reset_cam_params(self):
        self.renderer_kwargs.update({"azim": self.default_azim, "elev": self.default_elev, "dist": self.default_dist})
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
        img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img
        self.camera_scales["azim"].set(self.default_azim)
        self.camera_scales["elev"].set(self.default_elev)
        self.camera_scales["dist"].set(self.default_dist)

    def _zeros_to_concat(self):
        if self.model_type == "smplx":
            pass
        elif self.model_type == "flame":
            if self.with_face:
                return torch.zeros((40)).tolist()
            else:
                return torch.zeros(1, 90).tolist()
        else:
            return torch.zeros((31)).tolist()

    def create_application(self):

        # ------------------ Create the root window ------------------
        self.root = tkinter.Tk()
        self.root.title("Text 2 Mesh - PyTorch3D")
        if self.on_parameters:
            self.root.geometry("1000x1000")
        else:
            self.root.geometry("800x560")

        img_coords = (80, 10)

        parameters_main_frame = tkinter.Frame(self.root, bg="white", borderwidth=0)
        image_main_frame = tkinter.Frame(self.root, bg="white", borderwidth=0)

        img_frame = tkinter.Frame(
            self.root, highlightbackground="white", highlightthickness=0, bg="white", borderwidth=0
        )
        parameters_frame = tkinter.Frame(
            self.root, highlightbackground="white", highlightthickness=0, bg="white", borderwidth=0
        )
        parameters_main_frame.pack(fill=tkinter.BOTH, expand=True, side=tkinter.LEFT)
        image_main_frame.pack(fill=tkinter.BOTH, expand=True, side=tkinter.RIGHT)
        # ------------------------------------------------------------

        # ------------------------ Image ----------------------------
        img_canvas = tkinter.Canvas(image_main_frame, bg="white", highlightbackground="white", borderwidth=0)
        img_canvas.pack(fill=tkinter.BOTH, expand=True, side=tkinter.LEFT)
        img_canvas.create_window(img_coords, window=img_frame, anchor=tkinter.N)
        img = ImageTk.PhotoImage(image=self.img)
        self.img_label = tkinter.Label(img_frame, image=img, borderwidth=0)
        self.img_label.pack(fill=tkinter.BOTH, expand=True, side=tkinter.LEFT)
        # ------------------------------------------------------------

        # ----------------------- Parameters -------------------------
        self.parameters_canvas = tkinter.Canvas(
            parameters_main_frame, bg="white", highlightbackground="white", borderwidth=0
        )
        self.parameters_canvas.pack(side=tkinter.LEFT, padx=0, pady=0, anchor="nw")
        self.parameters_canvas.create_window((0, 0), window=parameters_frame, anchor=tkinter.NW)
        # ------------------------------------------------------------

        # ------------------- Parameters Scale Bars ------------------
        if self.on_parameters:

            if self.model_type == "smplx":
                scale_kwargs = self.get_parameters_scale_kwargs()
                if self.with_face:
                    for expression in range(self.expression.shape[1]):
                        expression_scale = tkinter.Scale(
                            parameters_frame,
                            label=f"expression {expression}",
                            command=self.update_expression(expression),
                            **scale_kwargs,
                        )
                        self.production_scales.append(expression_scale)
                        expression_scale.set(0)
                        expression_scale.pack()
                else:
                    for beta in range(self.betas.shape[1]):
                        betas_scale = tkinter.Scale(
                            parameters_frame,
                            label=f"beta {beta}",
                            command=self.update_betas(beta),
                            **scale_kwargs,
                        )
                        self.production_scales.append(betas_scale)
                        betas_scale.set(0)
                        betas_scale.pack()

            elif self.model_type == "flame":
                scale_kwargs = self.get_parameters_scale_kwargs()
                if self.with_face:
                    for label in range(self.face_expression.shape[1]):
                        label_tag = f"expression {label}" if label != self.face_expression.shape[1] - 1 else "jaw pose"
                        face_expression_scale = tkinter.Scale(
                            parameters_frame,
                            label=label_tag,
                            command=self.update_face_expression(label),
                            **scale_kwargs,
                        )
                        self.production_scales.append(face_expression_scale)
                        face_expression_scale.set(0)
                        face_expression_scale.pack()
                else:
                    scale_kwargs = self.get_parameters_scale_kwargs()
                    for label in range(self.face_shape.shape[1]):
                        face_shape_scale = tkinter.Scale(
                            parameters_frame,
                            label=f"shape param {label}",
                            command=self.update_face_shape(label),
                            **scale_kwargs,
                        )
                        self.production_scales.append(face_shape_scale)
                        face_shape_scale.set(0)
                        face_shape_scale.pack()

            else:
                scale_kwargs = self.get_smal_scale_kwargs()
                for label in range(self.beta.shape[1]):
                    beta_shape_scale = tkinter.Scale(
                        parameters_frame,
                        label=f"beta param {label}",
                        command=self.update_beta_shape(label),
                        **scale_kwargs,
                    )
                    self.production_scales.append(beta_shape_scale)
                    beta_shape_scale.set(0)
                    beta_shape_scale.pack()

        else:
            scale_kwargs = self.get_stats_scale_kwargs()
            for idx, (label, value) in enumerate(self.mean_values.items()):
                label_scale = tkinter.Scale(
                    parameters_frame,
                    label=label,
                    command=self.update_labels(idx),
                    **scale_kwargs,
                )
                self.production_scales.append(label_scale)
                label_scale.set(value)
                label_scale.pack()
        # ------------------------------------------------------------

        # --------------------- Camera Controls ----------------------
        zoom_scale_kwarg = self.get_zoom_scale_kwargs()
        zoom_in_scale = tkinter.Scale(
            parameters_frame,
            label="Zoom - in <-> out",
            command=lambda x: self.update_camera_zoom(x),
            **zoom_scale_kwarg,
        )
        zoom_in_scale.set(self.default_dist)
        zoom_in_scale.pack(pady=(50, 0))
        self.camera_scales["dist"] = zoom_in_scale

        azim_scale_kwarg = self.get_azim_scale_kwargs()
        azim_scale = tkinter.Scale(
            parameters_frame,
            label="azim - left <-> right",
            command=lambda x: self.update_camera_azim(x),
            **azim_scale_kwarg,
        )
        azim_scale.set(self.default_azim)
        azim_scale.pack()
        self.camera_scales["azim"] = azim_scale

        elev_scale_kwarg = self.get_elev_scale_kwargs()
        elev_scale = tkinter.Scale(
            parameters_frame,
            label="elev - down <-> up",
            command=lambda x: self.update_camera_elev(x),
            **elev_scale_kwarg,
        )
        elev_scale.set(self.default_elev)
        elev_scale.pack()
        self.camera_scales["elev"] = elev_scale
        # ------------------------------------------------------------

        # ------------------------ Scroll Bar ------------------------
        # adding Scrollbar
        y_scroll = tkinter.Scrollbar(
            img_frame,
            orient=tkinter.VERTICAL,
            command=self.parameters_canvas.yview,
            bg="white",
            troughcolor="white",
            activebackground="black",
            highlightbackground="white",
            width=8,
        )
        parameters_main_frame.bind(
            "<Configure>", lambda e: self.parameters_canvas.configure(scrollregion=self.parameters_canvas.bbox("all"))
        )
        self.parameters_canvas.configure(yscrollcommand=y_scroll.set)
        y_scroll.pack(
            fill=tkinter.Y,
            side=tkinter.LEFT,
        )
        # ------------------------------------------------------------

        # ------------------------ Buttons --------------------------
        reset_button_kwargs = self.get_reset_button_kwargs()

        # all reset button
        reset_all_button = tkinter.Button(
            parameters_frame, text="Reset All", command=lambda: self.reset_parameters(), **reset_button_kwargs
        )
        reset_all_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP, pady=(50, 0))

        # reset camera button
        reset_camera_button = tkinter.Button(
            parameters_frame, text="Reset Camera Params", command=lambda: self.reset_cam_params(), **reset_button_kwargs
        )
        reset_camera_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # add texture button
        add_texture_button = tkinter.Button(
            parameters_frame, text="Add Texture", command=lambda: self.add_texture(), **reset_button_kwargs
        )
        add_texture_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # remove texture button
        remove_texture_button = tkinter.Button(
            parameters_frame, text="Remove Texture", command=lambda: self.remove_texture(), **reset_button_kwargs
        )
        remove_texture_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # save image & params button
        save_img_n_params_button = tkinter.Button(
            parameters_frame, text="save png", command=lambda: self.save_png(), **reset_button_kwargs
        )
        save_img_n_params_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # save image & params button
        save_obj_button = tkinter.Button(
            parameters_frame, text="save obj", command=lambda: self.save_obj(), **reset_button_kwargs
        )
        save_obj_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # generate random params button
        random_params_button = tkinter.Button(
            parameters_frame, text="random params", command=lambda: self.random_button(), **reset_button_kwargs
        )
        random_params_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)
        # ------------------------------------------------------------
        self.parameters_canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.root.mainloop()

    def get_zoom_scale_kwargs(self) -> Dict[str, Any]:
        if self.model_type == "flame":
            return {
                "from_": 0.45,
                "to": 1,
                "resolution": 0.01,
                "orient": tkinter.HORIZONTAL,
                "length": 200,
                "bg": "white",
                "highlightbackground": "white",
                "highlightthickness": 0,
                "troughcolor": "pink",
                "width": 4,
            }
        else:
            return {
                "from_": 2.0,
                "to": 10.0,
                "resolution": 0.1,
                "orient": tkinter.HORIZONTAL,
                "length": 200,
                "bg": "white",
                "highlightbackground": "white",
                "highlightthickness": 0,
                "troughcolor": "pink",
                "width": 4,
            }

    @staticmethod
    def get_azim_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": -180,
            "to": 180,
            "resolution": 0.1,
            "orient": tkinter.HORIZONTAL,
            "length": 200,
            "bg": "white",
            "highlightbackground": "white",
            "highlightthickness": 0,
            "troughcolor": "pink",
            "width": 4,
        }

    @staticmethod
    def get_elev_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": -90,
            "to": 90,
            "resolution": 0.1,
            "orient": tkinter.HORIZONTAL,
            "length": 200,
            "bg": "white",
            "highlightbackground": "white",
            "highlightthickness": 0,
            "troughcolor": "pink",
            "width": 4,
        }

    @staticmethod
    def get_smal_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": -5,
            "to": 5,
            "resolution": 0.01,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "black",
            "width": 4,
            "length": 200,
            "borderwidth": 0,
        }

    @staticmethod
    def get_parameters_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": -5,
            "to": 5,
            "resolution": 0.01,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "pink",
            "width": 4,
            "length": 200,
            "borderwidth": 0,
        }

    @staticmethod
    def get_stats_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": 0,
            "to": 50,
            "resolution": 1,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "pink",
            "width": 4,
            "length": 200,
            "borderwidth": 0,
            "highlightbackground": "white",
            "highlightthickness": 0,
        }

    @staticmethod
    def get_reset_button_kwargs() -> Dict[str, Any]:
        return {
            "activebackground": "black",
            "activeforeground": "white",
            "bg": "white",
            "fg": "black",
            "highlightbackground": "white",
        }


@hydra.main(config_path="../config", config_name="sliders_demo_py3d")
def main(cfg):
    app = SlidersApp(cfg.demo_kwargs)
    app.create_application()


if __name__ == "__main__":
    main()
