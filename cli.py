#!/usr/bin/env python3
"""FST 工具统一命令行入口"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='FST 车位分类器工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 标注工具
    label_parser = subparsers.add_parser('label', help='启动标注工具')
    label_parser.add_argument('--images', default='dataset/images', help='图片目录')
    label_parser.add_argument('--labels', default='dataset/labels', help='标签目录')
    label_parser.add_argument('--auto', action='store_true', help='使用半自动标注')

    # 训练
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data', default='./dataset', help='数据目录')
    train_parser.add_argument('--output', default='./checkpoints', help='输出目录')
    train_parser.add_argument('--backbone', default='efficientnet_b0', help='骨干网络')
    train_parser.add_argument('--batch', type=int, default=32, help='批次大小')
    train_parser.add_argument('--epochs1', type=int, default=10, help='阶段1轮数')
    train_parser.add_argument('--epochs2', type=int, default=30, help='阶段2轮数')

    # 导出
    export_parser = subparsers.add_parser('export', help='导出 ONNX 模型')
    export_parser.add_argument('--checkpoint', default='checkpoints/best_model.pth', help='模型检查点')
    export_parser.add_argument('--output', default='fst_classifier.onnx', help='输出文件')

    # 推理
    infer_parser = subparsers.add_parser('infer', help='运行推理')
    infer_parser.add_argument('--image', help='单张图片路径')
    infer_parser.add_argument('--images', help='批量图片目录')
    infer_parser.add_argument('--model', default='checkpoints/best_model.pth', help='模型路径')
    infer_parser.add_argument('--output', default='./results', help='输出目录')
    infer_parser.add_argument('--visualize', action='store_true', help='可视化结果')

    # Web 界面
    app_parser = subparsers.add_parser('app', help='启动 Gradio 界面')
    app_parser.add_argument('--model', default='checkpoints/best_model.pth', help='模型路径')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 执行命令
    if args.command == 'label':
        if args.auto:
            from fst.auto_label_tool import main as auto_label_main
            sys.argv = ['auto_label_tool', '--images', args.images, '--labels', args.labels]
            auto_label_main()
        else:
            from fst.label_tool import main as label_main
            sys.argv = ['label_tool', '--images', args.images, '--labels', args.labels]
            label_main()

    elif args.command == 'train':
        from fst.train import main as train_main
        sys.argv = ['train', '--data', args.data, '--output', args.output,
                    '--backbone', args.backbone, '--batch', str(args.batch),
                    '--epochs1', str(args.epochs1), '--epochs2', str(args.epochs2)]
        train_main()

    elif args.command == 'export':
        from fst.export_onnx import main as export_main
        sys.argv = ['export_onnx', '--checkpoint', args.checkpoint, '--output', args.output]
        export_main()

    elif args.command == 'infer':
        if args.image:
            from examples.hybrid_inference_demo import main as infer_main
            sys.argv = ['hybrid_inference_demo', '--image', args.image, '--model', args.model]
            if args.visualize:
                sys.argv.append('--visualize')
            infer_main()
        elif args.images:
            from examples.batch_inference import main as batch_main
            sys.argv = ['batch_inference', '--images', args.images, '--model', args.model,
                        '--output', args.output]
            if args.visualize:
                sys.argv.append('--visualize')
            batch_main()
        else:
            print('错误: 请指定 --image 或 --images')

    elif args.command == 'app':
        from fst.app import main as app_main
        sys.argv = ['app', '--model', args.model]
        app_main()

if __name__ == '__main__':
    main()
