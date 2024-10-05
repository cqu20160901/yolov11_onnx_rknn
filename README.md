# yolov11_onnx_rknn
yolov11 部署版本，将DFL放在后处理中，便于移植不同平台,后处理为C++部署而写，python 测试后处理时耗意义不大。


# 文件夹结构说明

yolov11n_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolov8n_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本(使用的版本rknn_toolkit2-2.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)

# 测试结果

pytorch结果

![image](https://github.com/cqu20160901/yolov11_onnx_rknn/blob/main/yolov11n_onnx/test_pytorch_result.jpg)

onnx 结果

![image](https://github.com/cqu20160901/yolov11_onnx_rknn/blob/main/yolov11n_onnx/test_onnx_result.jpg)


# rk3588 部署结果

[rk3588 C++部署代码参考链接](https://github.com/cqu20160901/yolov11_dfl_rknn_Cplusplus)

![image](https://github.com/cqu20160901/yolov11_dfl_rknn_Cplusplus/blob/main/examples/rknn_yolov11_demo_dfl_open/test_result.jpg)

时耗
![image](https://github.com/cqu20160901/yolov11_dfl_rknn_Cplusplus/blob/main/examples/rknn_yolov11_demo_dfl_open/yolov11_rk3588_costtime.png)
