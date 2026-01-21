#!/usr/bin/env python3
"""
QAGen Pipeline 主调度脚本
依次调用三个步骤：OCR -> 建图 -> 问题生成

支持 --domain 参数指定行业/领域子目录
"""

import sys
import subprocess
import argparse
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).parent


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def list_available_domains(working_dir: Path) -> dict:
    """列出各目录下可用的domains"""
    result = {
        "raw": [],
        "processed": [],
        "lightrag_db": [],
        "output": [],
    }
    
    for key in result:
        dir_path = working_dir / key
        if dir_path.exists():
            domains = [d.name for d in dir_path.iterdir() if d.is_dir()]
            result[key] = sorted(domains)
    
    return result


def run_step(step_name: str, script_path: Path, extra_args: list = None) -> bool:
    """
    运行单个步骤
    
    Args:
        step_name: 步骤名称
        script_path: 脚本路径
        extra_args: 额外参数
        
    Returns:
        是否成功
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"运行: {step_name}")
    logger.info(f"{'=' * 60}")

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"✅ {step_name} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} 失败 (退出码: {e.returncode})")
        return False
    except Exception as e:
        logger.error(f"❌ {step_name} 异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="QAGen Pipeline - 问题生成流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整流水线 (处理所有数据)
  python main.py

  # 指定 domain 运行流水线
  python main.py --domain Robot
  python main.py -d Numerical-Control-System

  # 仅运行步骤1 (OCR) 指定domain
  python main.py --step 1 --domain Robot

  # 运行步骤2和3 (已有OCR文本)
  python main.py --step 2 --step 3 --domain Robot

  # 列出可用的 domains
  python main.py --list-domains

  # 单独运行各步骤
  python steps/1_pdf_to_text.py --domain Robot
  python steps/2_build_graph.py --domain Robot
  python steps/3_gen_questions.py --domain Robot

目录结构 (使用 --domain):
  working/
  ├── raw/
  │   ├── Robot/                # --domain Robot
  │   │   └── *.pdf
  │   └── Numerical-Control-System/
  │       └── *.pdf
  ├── processed/
  │   ├── Robot/
  │   │   └── *.txt, *.md
  │   └── ...
  ├── lightrag_db/
  │   ├── Robot/                # 每个domain独立的知识图谱
  │   │   └── (LightRAG files)
  │   └── ...
  └── output/
      ├── Robot/
      │   └── *_questions.jsonl
      └── ...
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--domain", "-d",
        type=str,
        help="指定行业/领域子目录 (如: Robot, Numerical-Control-System)"
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        action="append",
        choices=[1, 2, 3],
        help="指定运行的步骤 (可多次指定，默认运行全部)"
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="列出可用的domain目录"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    # 加载配置获取working_dir
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import load_settings
    settings = load_settings(args.config)
    paths = settings.get_working_paths(PROJECT_ROOT)

    # 列出可用domains
    if args.list_domains:
        domains_info = list_available_domains(paths["working_dir"])
        
        print("=" * 50)
        print("可用的 Domain 目录")
        print("=" * 50)
        
        print(f"\n📁 raw/ (PDF输入):")
        if domains_info["raw"]:
            for d in domains_info["raw"]:
                print(f"   - {d}")
        else:
            print("   (空)")
        
        print(f"\n📁 processed/ (OCR文本):")
        if domains_info["processed"]:
            for d in domains_info["processed"]:
                print(f"   - {d}")
        else:
            print("   (空)")
        
        print(f"\n📁 lightrag_db/ (知识图谱):")
        if domains_info["lightrag_db"]:
            for d in domains_info["lightrag_db"]:
                print(f"   - {d}")
        else:
            print("   (空)")
        
        print(f"\n📁 output/ (问题文件):")
        if domains_info["output"]:
            for d in domains_info["output"]:
                print(f"   - {d}")
        else:
            print("   (空)")
        
        print("")
        return

    # 确定要运行的步骤
    steps_to_run = args.step if args.step else [1, 2, 3]
    steps_to_run = sorted(set(steps_to_run))

    logger.info("=" * 60)
    logger.info("QAGen Pipeline - 问题生成流水线")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    if args.domain:
        logger.info(f"Domain: {args.domain}")
    logger.info(f"运行步骤: {steps_to_run}")
    logger.info("")

    # 步骤定义
    step_info = {
        1: ("步骤1: PDF转文本 (OCR)", PROJECT_ROOT / "steps" / "1_pdf_to_text.py"),
        2: ("步骤2: 构建知识图谱 (LightRAG)", PROJECT_ROOT / "steps" / "2_build_graph.py"),
        3: ("步骤3: 生成问题", PROJECT_ROOT / "steps" / "3_gen_questions.py"),
    }

    # 公共参数
    common_args = ["--config", str(args.config), "--log-level", args.log_level]
    if args.domain:
        common_args.extend(["--domain", args.domain])

    # 运行步骤
    success_count = 0
    failed_steps = []

    for step_num in steps_to_run:
        step_name, script_path = step_info[step_num]

        if not script_path.exists():
            logger.error(f"脚本不存在: {script_path}")
            failed_steps.append(step_num)
            continue

        if run_step(step_name, script_path, common_args):
            success_count += 1
        else:
            failed_steps.append(step_num)
            # 如果某一步失败，询问是否继续
            if step_num != steps_to_run[-1]:
                logger.warning(f"步骤{step_num}失败，后续步骤可能依赖此步骤的输出")

    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("流水线执行完成")
    logger.info("=" * 60)
    if args.domain:
        logger.info(f"Domain: {args.domain}")
    logger.info(f"成功: {success_count}/{len(steps_to_run)} 步骤")
    
    if failed_steps:
        logger.error(f"失败步骤: {failed_steps}")
        sys.exit(1)
    else:
        logger.info("✅ 所有步骤执行成功！")
        logger.info("")
        if args.domain:
            logger.info(f"输出文件位于 working/output/{args.domain}/ 目录")
        else:
            logger.info("输出文件位于 working/output/ 目录")


if __name__ == "__main__":
    main()
