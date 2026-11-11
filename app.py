"""Stream video through tiled Real-ESRGAN without holding a clip in memory."""
import functools
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from runtime.common import asset_path, device, guarded, output_dir, output_path, page_header

MODES = {"通用快速": "main", "照片高细节（较慢）": "detail", "动漫": "anime"}


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def video_info(path, output=False):
    info = probe(path)
    streams = [s for s in info["streams"] if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")]
    if len(streams) != 1:
        raise ValueError("请上传只有一条视频画面的文件")
    stream = streams[0]
    # Reject rotation and anamorphic input rather than silently changing its aspect ratio.
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    rotation += sum(float(s.get("rotation", 0)) for s in stream.get("side_data_list", []))
    if rotation % 360 or stream.get("sample_aspect_ratio", "1:1") not in ("1:1", "0:1", "N/A"):
        raise ValueError("请先把视频导出为正确方向、方形像素的MP4，再上传")
    fps = Fraction(stream.get("avg_frame_rate", "0/1"))
    rate = Fraction(stream.get("r_frame_rate", "0/1"))
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
    width, height = int(stream["width"]), int(stream["height"])
    if not 0 < fps <= 60 or not math.isfinite(duration) or not 0 < duration <= 60.05:
        raise ValueError("支持最长60秒、最高60FPS的视频；请先裁剪后上传")
    if rate <= 0 or abs(float(rate / fps) - 1) > 0.001:
        raise ValueError("请先导出为固定帧率MP4；当前不支持可变帧率视频")
    if not output and (width * height > 1920 * 1080 or min(width, height) < 16):
        raise ValueError("输入最多1920×1080像素；请先降低输入分辨率")
    if stream.get("color_transfer") in ("smpte2084", "arib-std-b67"):
        raise ValueError("当前支持SDR视频，请先把HDR素材转换为SDR")
    return {"width": width, "height": height, "fps": fps, "duration": duration,
            "audio": any(s["codec_type"] == "audio" for s in info["streams"])}


@functools.lru_cache(maxsize=1)
def model(kind):
    import torch
    if kind == "detail":
        from network import RRDBNet
        net = RRDBNet()
    else:
        from compact import SRVGGNetCompact
        net = SRVGGNetCompact(16 if kind == "anime" else 32)
    weights = torch.load(asset_path(kind), map_location="cpu", weights_only=True)
    net.load_state_dict(weights.get("params_ema", weights.get("params", weights)), strict=True)
    net = net.eval().to(device())
    return net.half() if device() == "cuda" else net


def upscale_frame(frame, net, scale):
    import numpy as np
    import torch
    from PIL import Image
    height, width = frame.shape[:2]
    result = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
    tile, border = 256, 32
    dtype = next(net.parameters()).dtype
    with torch.inference_mode():
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                ye, xe = min(y + tile, height), min(x + tile, width)
                y0, x0 = max(0, y - border), max(0, x - border)
                y1, x1 = min(height, ye + border), min(width, xe + border)
                data = np.ascontiguousarray(frame[y0:y1, x0:x1])
                tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(device(), dtype=dtype) / 255
                prediction = net(tensor).clamp(0, 1)[0].float().permute(1, 2, 0).cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise RuntimeError("模型产生非有限结果，请重启服务后重试")
                pixels = np.rint(prediction * 255).astype(np.uint8)
                result[y*4:ye*4, x*4:xe*4] = pixels[(y-y0)*4:(ye-y0)*4, (x-x0)*4:(xe-x0)*4]
    if scale == 2:
        result = np.asarray(Image.fromarray(result).resize((width*2, height*2), Image.Resampling.LANCZOS))
    return result


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def read_frame(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    if data and len(data) != size:
        raise RuntimeError("视频解码得到不完整帧，任务已停止")
    return data


@guarded
def upscale_video(source, mode, scale, progress=None):
    import numpy as np
    import torch
    if not source or not Path(source).is_file():
        raise ValueError("请先上传视频")
    if mode not in MODES or scale not in (2, 4):
        raise ValueError("请选择有效的模式和放大倍数")
    info = video_info(source)
    width, height = info["width"], info["height"]
    ow, oh = width * int(scale), height * int(scale)
    if ow * oh > 3840 * 2160 or max(ow, oh) > 4096:
        raise ValueError("输出最多4K，请改选2倍或先降低输入分辨率")
    if not torch.cuda.is_available():
        raise ValueError("视频超分请使用有GPU模式启动实例")
    if shutil.disk_usage(output_dir()).free < 2_000_000_000:
        raise ValueError("输出盘剩余空间不足2GB，请先清理已有输出")
    net = model(MODES[mode])
    destination = output_path(".mp4")
    decoder = encoder = None
    count = 0
    start = time.monotonic()
    fps = info["fps"]
    estimate = max(1, round(info["duration"] * float(fps)))
    success = False
    with tempfile.TemporaryDirectory(prefix="video-sr-", dir=output_dir()) as work:
        silent = Path(work) / "video.mp4"
        with (Path(work)/"decode.log").open("w+b") as dec_log, (Path(work)/"encode.log").open("w+b") as enc_log:
            try:
                decoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-i", str(source),
                    "-map", "0:v:0", "-an", "-sn", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=dec_log)
                encoder = subprocess.Popen(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent)], stdin=subprocess.PIPE, stderr=enc_log)
                while data := read_frame(decoder.stdout, width * height * 3):
                    if count >= math.ceil(60.1 * float(fps)):
                        raise ValueError("解码帧数超过60秒限制，请先裁剪视频")
                    frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
                    result = upscale_frame(frame, net, int(scale))
                    encoder.stdin.write(result.tobytes())
                    count += 1
                    if progress:
                        elapsed = time.monotonic() - start
                        progress(min(count / estimate, 0.99), desc=f"已处理{count}/{estimate}帧 · {elapsed:.0f}秒")
                decoder.stdout.close()
                if decoder.wait(timeout=30):
                    raise RuntimeError("视频解码失败，请检查输入文件")
                encoder.stdin.close()
                if encoder.wait(timeout=90) or not count:
                    raise RuntimeError("视频编码失败，未产生有效输出")
                # Re-encode the first audio track to AAC for MP4/browser compatibility.
                subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(silent), "-i", str(source),
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{count / float(fps):.9f}", "-map_metadata", "-1", "-movflags", "+faststart", str(destination)],
                    check=True, capture_output=True, timeout=120)
                result_info = video_info(destination, output=True)
                if result_info["width"] != ow or result_info["height"] != oh:
                    raise RuntimeError("输出尺寸校验失败")
                if info["audio"] and not result_info["audio"]:
                    raise RuntimeError("输出音轨校验失败")
                if abs(result_info["duration"] - count / float(fps)) > 0.1:
                    raise RuntimeError("输出时长校验失败")
                elapsed = time.monotonic() - start
                summary = (f"完成：{width}×{height} → {ow}×{oh}，{float(fps):g}FPS，{count}帧，"
                           f"{count/float(fps):.2f}秒；{'已保留音轨' if info['audio'] else '原视频无音轨'}。耗时{elapsed:.1f}秒。")
                success = True
                return str(destination), str(destination), summary
            finally:
                stop_process(decoder)
                stop_process(encoder)
                if not success:
                    destination.unlink(missing_ok=True)


def build_ui():
    import gradio as gr
    def run(source, mode, scale, progress=gr.Progress()):
        return upscale_video(source, mode, scale, progress)
    with gr.Blocks(title="Real-ESRGAN 视频超分") as ui:
        page_header()
        gr.Markdown("上传视频 → 选择模式 → 下载MP4。默认通用快速；照片高细节较慢，动漫素材建议选动漫。\n\n"
                    "支持200MB内、最长60秒的SDR固定帧率视频，输入最多1080P，输出最多4K；保留第一条音轨。"
                    "采用逐帧增强，可能保留或放大闪烁，不能保证恢复真实细节。")
        source = gr.File(label="上传视频", file_types=[".mp4", ".mov", ".mkv", ".webm"], type="filepath")
        mode = gr.Radio(list(MODES), value="通用快速", label="增强模式")
        scale = gr.Radio([2, 4], value=2, label="宽高放大倍数")
        button = gr.Button("开始视频超分", variant="primary")
        result = gr.Video(label="超分结果", format="mp4")
        download = gr.File(label="下载MP4")
        status = gr.Textbox(label="处理结果", interactive=False)
        button.click(run, [source, mode, scale], [result, download, status], api_name="upscale_video")
    return ui
