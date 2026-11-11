# 视频超分方案选择（2026-11-12）

1. 官方Real-ESRGAN提供视频推理入口和general-x4v3、animevideov3、x4plus模型：<https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py>、<https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py>。
2. SRVGG网络层数和参数键由官方架构及推理入口核对：<https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py>。
3. v0.2.5.0官方发布记录显示通用模型4,885,111字节，动漫模型2,504,012字节：<https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0>。x4plus为67,040,989字节：<https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0>。

选择原因：利用现有经验证的PyTorch环境与RRDB权重，增加通用快速和动漫模型，仅增加约7.4MB权重。三种模式覆盖常见视频素材，避免在剩余2.34GB镜像额度下引入新的大型依赖栈。该选择不代表在所有视频评测上质量最优；它是轻量、可离线分发的实用方案。与BasicVSR++等时序超分不同，本应用没有多帧运动补偿，不宣传为防闪烁模型。
