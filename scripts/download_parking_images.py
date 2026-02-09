"""
停车位图片批量采集脚本.

从多个免费/开放图源搜索并下载"地面视角停车位"照片。
下载后需要人工清洗（删除不符合要求的图片）。

用法:
    pip install requests beautifulsoup4 icrawler
    python download_parking_images.py --output ./raw_images --count 500

依赖:
    pip install icrawler requests pillow
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path


def download_with_icrawler(output_dir: str, count_per_query: int = 100):
    """
    使用 icrawler 从 Bing/Google 图片搜索下载.
    icrawler 是一个轻量图片爬虫库，支持多个搜索引擎。
    """
    try:
        from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    except ImportError:
        print("请先安装 icrawler: pip install icrawler")
        return

    # ── 搜索关键词（中英文混合，覆盖各种场景）──
    queries_zh = [
        "地下车库车位 空车位",
        "停车场车位 手机拍摄",
        "地下停车场 空车位 近拍",
        "室内停车场 车位线",
        "地面停车场 白线车位",
        "车位 黄线 地下车库",
        "垂直车位 地面拍摄",
        "侧方停车位 路边",
        "小区停车场 车位",
        "商场地下车库 空车位",
        "车位 路沿 柱子",
        "断头路车位 停车",
        "砖草车位 停车场",
        "微型车位 地下车库",
        "车位 锥桶 占用",
        "停车位 水马 障碍物",
        "鱼骨车位 斜列式",
    ]

    queries_en = [
        "empty parking spot ground view",
        "parking space ground level photo",
        "underground parking lot empty spot",
        "parking bay white lines close up",
        "indoor parking garage empty space",
        "parking spot with curb ground view",
        "parallel parking space street view",
        "angled parking spot photo",
        "parking space between two cars",
        "parking spot with pillar obstacle",
        "parking lot cone barrier",
        "narrow parking space garage",
        "parking spot green floor garage",
        "parking space yellow line marking",
    ]

    all_queries = queries_zh + queries_en
    os.makedirs(output_dir, exist_ok=True)

    for i, query in enumerate(all_queries):
        print(f"\n[{i+1}/{len(all_queries)}] Searching: {query}")
        sub_dir = os.path.join(output_dir, f"batch_{i:03d}")
        os.makedirs(sub_dir, exist_ok=True)

        try:
            # 用 Bing（比 Google 限制少）
            crawler = BingImageCrawler(
                storage={"root_dir": sub_dir},
                downloader_threads=4,
            )
            crawler.crawl(
                keyword=query,
                max_num=count_per_query,
                min_size=(300, 300),  # 过滤太小的图
            )
        except Exception as e:
            print(f"  [WARN] Failed: {e}")

        # 限速，避免被封
        time.sleep(2)

    print(f"\n下载完成! 图片保存在: {output_dir}")
    print(f"请手动清洗：删除不是地面视角停车位的图片")


def flatten_and_dedup(output_dir: str, final_dir: str):
    """
    将所有子文件夹的图片合并到一个目录，并按 MD5 去重。
    """
    os.makedirs(final_dir, exist_ok=True)
    seen_hashes = set()
    count = 0

    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue
            src = os.path.join(root, f)
            # 计算 MD5 去重
            with open(src, "rb") as fh:
                h = hashlib.md5(fh.read()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # 重命名为统一格式
            ext = Path(f).suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"
            dst = os.path.join(final_dir, f"parking_{count:05d}{ext}")
            os.rename(src, dst)
            count += 1

    print(f"去重完成: {count} 张唯一图片 -> {final_dir}")


def print_manual_sources():
    """打印手动下载资源列表."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              停车位图片获取渠道（手动 + 自动）                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  【最推荐】自己去停车场拍 (2-3小时 = 500张)                       ║
║    - 地下车库 2-3个                                              ║
║    - 地面停车场 1-2个                                            ║
║    - 路边侧方车位                                                ║
║    - 每个车位正面拍一张，注意覆盖各种类型                         ║
║                                                                  ║
║  【图片搜索引擎】                                                ║
║    - Bing图片搜索（本脚本自动爬取）                               ║
║    - 百度图片: image.baidu.com                                   ║
║      搜索: "地下车库车位" "空车位" "停车位近拍"                    ║
║    - Google Images (需梯子)                                      ║
║                                                                  ║
║  【短视频/社交平台截图】                                          ║
║    - 小红书搜索: "车位" "停车场" "微型车位"                       ║
║    - 抖音/快手搜索: "停车位" "地下车库"                           ║
║    - B站: 自动泊车测试视频 → 逐帧截图                            ║
║    - YouTube: "parking spot" "garage parking"                    ║
║                                                                  ║
║  【视频逐帧提取（量最大）】                                       ║
║    找 APA/自动泊车路测视频，用 ffmpeg 每隔 N 帧截一张:            ║
║    ffmpeg -i video.mp4 -vf "fps=1" frames/frame_%05d.jpg        ║
║                                                                  ║
║  【免费图库】                                                     ║
║    - Unsplash: unsplash.com/s/photos/parking-spot                ║
║    - Pexels: pexels.com/search/parking%20space/                  ║
║    - Pixabay: pixabay.com/images/search/parking%20lot/           ║
║                                                                  ║
║  【学术数据集（视角不完全匹配，但可参考）】                        ║
║    - Tongji ps2.0: cslinzhang.github.io/deepps (环视鸟瞰)        ║
║    - HuggingFace: TrainingDataPro/parking-space-detection        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(description="批量下载停车位训练图片")
    parser.add_argument("--output", default="./raw_images", help="下载目录")
    parser.add_argument("--final", default="./dataset/images", help="去重后最终目录")
    parser.add_argument("--count", type=int, default=50, help="每个关键词下载数量")
    parser.add_argument("--sources", action="store_true", help="只显示数据源列表")
    parser.add_argument("--dedup-only", action="store_true", help="只做去重合并")
    args = parser.parse_args()

    if args.sources:
        print_manual_sources()
        return

    if args.dedup_only:
        flatten_and_dedup(args.output, args.final)
        return

    print_manual_sources()
    print("开始自动爬取...\n")
    download_with_icrawler(args.output, count_per_query=args.count)
    flatten_and_dedup(args.output, args.final)


if __name__ == "__main__":
    main()
