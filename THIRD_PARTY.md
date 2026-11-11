# 模型与依赖

- Real-ESRGAN模型、SRVGG架构：<https://github.com/xinntao/Real-ESRGAN>，BSD-3-Clause，保留 `licenses/Real-ESRGAN.LICENSE.txt`。
- RRDB网络架构参考BasicSR：<https://github.com/XPixelGroup/BasicSR>，Apache-2.0，保留 `licenses/BasicSR.LICENSE.txt`。
- FFmpeg为系统安装组件，发行版配置及许可通过 `ffmpeg -L` 查看；使用发行版libx264编码，未修改或另行捆绑FFmpeg二进制。
- Gradio、PyTorch、NumPy、Pillow等依赖采用各自许可证。

模型版本与官方发布下载地址固定在app.json。官方视频处理入口为 `inference_realesrgan_video.py`；本项目采用自己的流式FFmpeg管道与分块推理实现，不依赖安装BasicSR及其旧版torchvision接口。
