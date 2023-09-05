"""AnimeSR Engine Module"""
import os
from typing import Any, Tuple
import torch
import onnxruntime as ort
import numpy as np
from ortei import IORTEngine

__all__ = ["AnimeSREngine"]


class AnimeSREngine(IORTEngine):
    """AnimeSREngine"""

    def __init__(
        self,
        onnx_path: str,
        device_id: int,
        device_name: str = "cuda",
        model_batch=1,
        upscale_rate=4,
        input_height=1080,
        input_width=1920,
    ) -> None:
        if device_name == "cuda":
            assert torch.cuda.is_available(), "ERROR::CUDA not available."
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
        self.onnx_path = onnx_path
        self.device_name = device_name
        self.device_id = device_id
        self.io_shape = {
            "input_frames": [model_batch, 3 * 3, input_height, input_width],
            "output_frame": [
                model_batch,
                3,
                input_height * upscale_rate,
                input_width * upscale_rate,
            ],
            "output_state": [model_batch, 64, input_height, input_width],
        }
        self.providers = [
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_use_max_workspace": "1",
                    "cudnn_conv_algo_search": "HEURISTIC",  # HEURISTIC, EXHAUSTIVE
                    "do_copy_in_default_stream": True,
                },
            ),
        ]

        # init
        super().__init__()
        self._bind_model_io()

        # warm-up
        self.inference()

    def _init_members(self):
        # set variable
        onnx_fp32_path = self.onnx_path
        device_name = self.device_name
        device_id = self.device_id
        providers = self.providers

        # set session options
        self.session_options = ort.SessionOptions()
        self.session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self.session_options.enable_profiling = False
        self.session_options.log_severity_level = 3
        self.providers = providers

        # build engine
        self.ort_session = ort.InferenceSession(
            path_or_bytes=onnx_fp32_path,
            sess_options=self.session_options,
            providers=providers,
        )

        # set binding data
        self.io_binding = self.ort_session.io_binding()
        self.io_data_cpu = {
            key: np.zeros(self.io_shape[key], np.float32) for key in self.io_shape
        }
        self.io_data_ort = {
            key: ort.OrtValue.ortvalue_from_numpy(
                self.io_data_cpu[key], device_name, device_id
            )
            for key in self.io_data_cpu
        }

        # set input, output
        _b, _c, _h, _w = self.io_shape["input_frames"]
        self.input_data = np.zeros([_b, _h, _w, _c], np.uint8)
        _b, _c, _h, _w = self.io_shape["output_frame"]
        self.output_data = np.zeros([_b, _h, _w, _c], np.uint8)

    def _bind_model_io(self) -> None:
        io_binding = self.io_binding
        io_data_ort = self.io_data_ort

        # binding
        io_binding.bind_ortvalue_input("input_frames", io_data_ort["input_frames"])
        io_binding.bind_ortvalue_input("prev_output", io_data_ort["output_frame"])
        io_binding.bind_ortvalue_input("input_state", io_data_ort["output_state"])
        io_binding.bind_ortvalue_output("output_state", io_data_ort["output_state"])
        io_binding.bind_ortvalue_output("output_frame", io_data_ort["output_frame"])

    def get_output_data(self) -> Any:
        return self.output_data

    def set_input_data(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        data : tuple of 3 np.ndarrays
        """
        for _c in range(9):
            self.input_data[:, :, :, _c] = data[int(_c / 3)][:, :, int(_c % 3)]

    def convert_data2input(self) -> None:
        io_data_cpu = self.io_data_cpu
        input_data = self.input_data

        # manual work required.
        image_raw = np.divide(input_data, 255.0, dtype=np.float32)
        for _c in range(9):
            io_data_cpu["input_frames"][:, _c, :, :] = image_raw[:, :, :, _c]

    def move_host2device(self) -> None:
        # init
        io_data_cpu: dict = self.io_data_cpu
        io_data_ort: dict = self.io_data_ort

        # manual work required.
        io_data_ort["input_frames"].update_inplace(io_data_cpu["input_frames"])

    def inference(self):
        # init
        ort_session = self.ort_session
        io_binding: dict = self.io_binding

        # inference
        ort_session.run_with_iobinding(io_binding)

    def move_device2host(self):
        # init
        io_data_ort: dict = self.io_data_ort
        io_data_cpu: dict = self.io_data_cpu

        # manual work required.
        io_data_cpu["output_frame"][:] = io_data_ort["output_frame"].numpy()[:]

    def convert_output2data(self):
        # init
        io_data_cpu: dict = self.io_data_cpu
        result_images: np.ndarray = self.output_data

        # manual work required.
        output = np.clip(np.multiply(io_data_cpu["output_frame"], 255.0), 0, 255)
        for c in range(3):
            result_images[:, :, :, c] = output[:, c, :, :]
