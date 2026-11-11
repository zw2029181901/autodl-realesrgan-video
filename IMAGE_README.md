# Real-ESRGAN 视频超分

视频2倍或4倍放大，通用快速、照片高细节和动漫三种模式。保留第一条音轨，输出MP4。支持200MB内、最长60秒、最高60FPS的SDR固定帧率视频，输入最多1080P，输出最多4K。1080P请选2倍；4倍建议输入960×540以内。

## 基本环境

Ubuntu22.04、Python3.12、PyTorch2.8.0、CUDA12.8。需要GPU，建议8GB以上显存；本次在RTX3090验证。三份权重合计约74.4MB，已放入系统盘，保存镜像无需再下载。另预留5GB用于上传、临时文件和结果。

## 使用方法

启动实例后打开自定义服务6006，上传视频，选择模式和倍数，点击开始，下载MP4。建议先试2至5秒素材。采用逐帧超分，不插帧，不能保证消除帧间闪烁或恢复真实细节。

- 项目：https://github.com/zw2029181901/autodl-realesrgan-video
- 代码：`/root/autodl-active`
- 权重：`/root/autodl-active/models`
- 启动：`bash /root/autodl-active/run-service.sh`
- 模式说明、输入限制与模型许可见README.md、THIRD_PARTY.md与licenses。

## 环境验证

三模式真实HTTP推理已通过；音轨、音量、帧数、帧率、尺寸及解码检查通过。另验证30000/1001分数帧率与分块推理。记录见IMAGE_VALIDATION.json。

本应用没有另建客户实例复建；开机配置沿用此前U2Net保存镜像新实例验证通过的机制。上述短片验收不代表所有尺寸与长视频的耗时或质量保证。

镜像UUID：`image-c4b4ade6fc`，平台体积1.24GB。v1已提交审核，尚未获得公开上架确认。
