#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import logging
import argparse
from typing import Dict
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


def preprocess(img_path):
    img_src = cv2.imread(img_path)
    image = cv2.resize(img_src, (640, 640), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255.0
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype, max_num_images=None,
                 exact_batches=False):
        """
        Args:
            input: The input directory to read images from.
            shape: The tensor shape of the batch to prepare,
                either in NCHW or NHWC format.
            dtype: The (numpy) datatype to cast the batched data to.
            max_num_images: The maximum number of images to read from
                the directory.
            exact_batches: This defines how to handle a number of images that
                is not an exact multiple of the batch size. If false, it will
                pad the final batch with zeros to reach the batch size.
                If true, it will *remove* the last few images in excess of a
                batch size multiple, to guarantee batches are exact (useful
                for calibration).
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[
                1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0: self.num_images]
        print('')
        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])
        # Indices
        self.image_index = 0
        self.batch_index = 0

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it
        within a loop as: for batch, images in batcher.get_batch(): ... Or
        outside of a batch with the next() function.

        Returns:
            A generator yielding two items per iteration: a numpy array holding
             a batch of images, and the list of paths to the images loaded
             within this batch.
        """
        for _, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i] = preprocess(image)
            self.batch_index += 1
            yield batch_data, batch_images



class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache
        file, an image batcher doesn't need to be defined. :param
        image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(
            self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2. Get the next batch to
        use for calibration, as a list of device memory pointers. :param
        names: The names of the inputs, if useful to define the order of
        inputs. :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(
                self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation,
                             np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(
                self.cache_file))
            f.write(cache)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on
        the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 4 * (2 ** 30)  # 4 GB
        
        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path, input_shapes: Dict = None):
        """
        Parse the ONNX graph and create the corresponding TensorRT network
        definition. :param onnx_path: The path to the ONNX graph to load.

        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in
                  range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in
                   range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name,
                                                                    input.shape,
                                                                    input.dtype))
        for output in outputs:
            print(output.name, output.shape, output.dtype)
            log.info(
                "Output '{}' with shape {} and dtype {}".format(output.name,
                                                                output.shape,
                                                                output.dtype))
        profile = self.builder.create_optimization_profile()
        for input_name, param in input_shapes.items():
            min_shape = param['min_shape']
            opt_shape = param['opt_shape']
            max_shape = param['max_shape']
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        if self.config.add_optimization_profile(profile) < 0:
            log.warning(f'Invalid optimization profile {profile}.')

        # assert self.batch_size > 0
        # self.builder.max_batch_size = self.batch_size

    def create_engine(
            self,
            engine_path,
            precision,
            calib_input=None,
            calib_cache=None,
            calib_num_images=25000,
            calib_batch_size=8
    ):
        """
        Build the TensorRT engine and serialize it to disk.

        Args:
            engine_path: The path where to serialize the engine to.
            precision: The datatype to use for the engine, either 'fp32',
                'fp16' or 'int8'.
            calib_input: The path to a directory, holding the calibration
                images.
            calib_cache: The path where to write the calibration cache to,
                or if it already exists, load it from.
            calib_num_images: The maximum number of images to use for
                calibration.
            calib_batch_size: The batch size to use for the calibration
                process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in
                  range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning(
                    "FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning(
                    "INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(
                        inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(
                            calib_input,
                            calib_shape,
                            calib_dtype,
                            max_num_images=calib_num_images,
                            exact_batches=True
                        )
                    )


        engine = self.builder.build_engine(self.network, self.config)
        if engine is None:
            print("ERROR: Failed to build the TensorRT engine.")
            exit(1)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        

def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx, input_shapes=dict(
        input=dict(min_shape=[1, 3, 640, 640],
                   opt_shape=[2, 3, 640, 640],
                   max_shape=[4, 3, 640, 640])))
    builder.create_engine(
        args.engine,
        args.precision,
        args.calib_input,
        args.calib_cache,
        args.calib_num_images,
        args.calib_batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="./yolov11n.onnx", help="The input ONNX model file to load")
    parser.add_argument("--engine", default="./yolov11n.trt", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or "
             "'int8', default: 'fp16'",
    )
    parser.add_argument("--verbose", default=False,  help="Enable more verbose log output")
    parser.add_argument("--calib_input", default="./images", help="The directory holding images to use for calibration")
    parser.add_argument(
        "--calib_cache",
        default="./calibration",
        help="The file path for INT8 calibration cache to use, default: ./calibration.cache",
    )
    parser.add_argument(
        "--calib_num_images",
        default=8,
        type=int,
        help="The maximum number of images to use for calibration, default: "
             "8",
    )
    parser.add_argument(
        "--calib_batch_size", default=2, type=int,
        help="The batch size for the calibration process, default: 1"
    )
    args = parser.parse_args()
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not any(
            [args.calib_input, args.calib_cache]):
        parser.print_help()
        log.error(
            "When building in int8 precision, either --calib_input or "
            "--calib_cache are required")
        sys.exit(1)
    main(args)