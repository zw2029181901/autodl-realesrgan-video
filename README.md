# Real-ESRGAN 视频超分

上传低清视频，选择通用快速、照片高细节或动漫模式，将宽高放大2倍或4倍，输出带音轨的MP4。默认通用快速。照片高细节计算量更大；动漫模式针对动画素材。

支持最长60秒、200MB、最高60FPS的SDR固定帧率视频，输入最高1920×1080像素，输出最多3840×2160像素（竖屏等同像素预算，最长边4096）。1080P输入请选择2倍；4倍建议输入不超过960×540。保留第一条音轨并编码为AAC；不保留字幕、额外音轨与HDR信息。旋转或非方形像素素材需要先导出为正确方向的普通MP4。

逐帧读取、分块推理与编码，不把整段视频或全部超分帧放进内存。不插帧，保留输入帧率。采用逐帧超分，不含运动补偿或时序一致性模型，可能保留或放大闪烁，不保证恢复真实细节。

## 镜像使用

镜像与审核状态见 `IMAGE_VALIDATION.json`。创建实例后打开自定义服务6006。需要GPU，建议8GB以上显存；源码安装时的基础环境与独立依赖空间另计，3个权重合计约74.4MB。建议预留5GB用于上传、临时视频与输出。

- 代码：`/root/autodl-active`
- 权重：`/root/autodl-active/models`，保存镜像包含权重。
- 启动：`bash /root/autodl-active/run-service.sh`
- Python3.12、PyTorch2.8.0、CUDA12.8。
- 推荐先用2至5秒视频确认效果和耗时，再处理长视频。

## 从源码安装

```bash
git clone https://github.com/zw2029181901/autodl-realesrgan-video.git
cd autodl-realesrgan-video
bash install.sh
.venv/bin/python runtime/prefetch.py --project .
bash start.sh
```

安装脚本安装FFmpeg；模型首次下载需要网络。模型和网络架构来源及许可证见 `THIRD_PARTY.md`、`licenses/`。本项目提供中文交互、流式处理、输入检查和部署配置，上游模型算法归原作者。
