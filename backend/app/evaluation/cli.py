import argparse
import sys
from pathlib import Path

from app.config.settings import get_settings
from app.core.container import build_container
from app.evaluation.report import write_report
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import EvaluationDataError, load_dataset
from app.persistence.database import build_engine, build_session_factory, initialize_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Roborock RAG 离线评测。")
    parser.add_argument("--dataset", required=True, help="JSONL 评测集路径。")
    parser.add_argument(
        "--mode",
        choices=("retrieval", "end-to-end"),
        default="retrieval",
        help="retrieval 只评测检索；end-to-end 额外调用生成模型。",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/baselines",
        help="JSON 与 Markdown 报告输出目录。",
    )
    parser.add_argument("--prefix", help="报告文件名前缀，默认由数据集和模式生成。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    if not settings.embedding_api_key:
        print("评测失败：尚未配置 Embedding API Key。", file=sys.stderr)
        return 2
    if args.mode == "end-to-end" and not settings.deepseek_api_key:
        print("评测失败：端到端模式尚未配置生成模型 API Key。", file=sys.stderr)
        return 2

    engine = build_engine(settings.database_url)
    try:
        initialize_database(engine)
        session_factory = build_session_factory(engine)
        container = build_container(settings, session_factory)
        container.indexing.recover_interrupted_tasks()
        cases = load_dataset(args.dataset)
        report = EvaluationRunner(
            settings=settings,
            documents=container.documents,
            retriever=container.retriever,
            chat=container.chat,
        ).run(cases, dataset_path=args.dataset, mode=args.mode)
        prefix = args.prefix or f"{Path(args.dataset).stem}_{args.mode.replace('-', '_')}"
        json_path, markdown_path = write_report(report, args.output_dir, prefix)
    except (EvaluationDataError, ValueError) as exc:
        print(f"评测失败：{exc}", file=sys.stderr)
        return 2
    finally:
        engine.dispose()

    print(f"评测完成：{json_path}")
    print(f"摘要报告：{markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
