import json
from pathlib import Path


def write_report(report: dict, output_dir: str | Path, prefix: str) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{prefix}.json"
    markdown_path = target_dir / f"{prefix}.md"
    json_payload = json.dumps(report, ensure_ascii=False, indent=2)
    if "api_key" in json_payload.casefold():
        raise ValueError("评测报告中禁止写入 API Key 字段。")
    json_path.write_text(json_payload + "\n", encoding="utf-8")
    markdown_path.write_text(_build_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _build_markdown(report: dict) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    test_metrics = summary["retrieval"]["test"]
    refusal = summary["refusal"]["test"]
    calibration = summary["threshold_calibration"]
    recommendation = calibration["recommendation"]
    lines = [
        "# Roborock RAG 评测报告",
        "",
        f"- 数据集：`{metadata['dataset']}`",
        f"- 评测模式：`{metadata['mode']}`",
        (
            f"- 样本数量：{metadata['case_count']}"
            f"（校准 {metadata['calibration_count']}，测试 {metadata['test_count']}）"
        ),
        f"- Embedding 模型：`{metadata['configuration']['embedding_model']}`",
        f"- 生成模型：`{metadata['configuration']['generation_model']}`",
        "",
        "## 测试集检索指标",
        "",
        "| 通道 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Page Recall@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for channel in ("dense", "bm25", "fused", "context"):
        metrics = test_metrics[channel]
        lines.append(
            "| {channel} | {h1} | {h3} | {h5} | {mrr} | {recall} |".format(
                channel=channel,
                h1=_percent(metrics.get("hit_at_1", 0.0)),
                h3=_percent(metrics.get("hit_at_3", 0.0)),
                h5=_percent(metrics.get("hit_at_5", 0.0)),
                mrr=f"{metrics.get('mrr_at_5', 0.0):.3f}",
                recall=_percent(metrics.get("page_recall_at_5", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## 拒答与阈值",
            "",
            f"- 测试集可回答问题放行率：{_percent(refusal['answer_recall'])}",
            f"- 测试集检索门控拒答率：{_percent(refusal['refusal_accuracy'])}",
            f"- 测试集生成证据门控样本数：{refusal['generation_refusal_case_count']}",
            f"- 当前 Dense distance 阈值：`{metadata['configuration']['max_dense_distance']}`",
            f"- 当前 BM25 阈值：`{metadata['configuration']['min_bm25_score']}`",
            f"- 校准集候选 Dense distance 阈值：`{recommendation['max_dense_distance']}`",
            f"- 校准集候选 BM25 阈值：`{recommendation['min_bm25_score']}`",
            f"- 独立测试结论：{calibration['holdout_validation']['decision']}",
            "",
        ]
    )
    if "generation" in summary:
        generation = summary["generation"]
        lines.extend(
            [
                "## 端到端回答指标",
                "",
                f"- 引用格式通过率：{_percent(generation['citation_validation_rate'])}",
                f"- 引用正确页命中率：{_percent(generation['citation_gold_hit_rate'])}",
                f"- 平均引用精确率：{_percent(generation['average_citation_precision'])}",
                f"- 平均关键事实覆盖率：{_percent(generation['average_keyword_coverage'])}",
                f"- 结构化拒答准确率：{_percent(generation['structured_refusal_accuracy'])}",
                f"- 自然语言拒答识别率：{_percent(generation['semantic_refusal_rate'])}",
                f"- 未结构化自然语言拒答数：{generation['unstructured_refusal_count']}",
                f"- 引用校验失败数：{generation['invalid_citation_count']}",
                f"- 端到端成功率：{_percent(generation['end_to_end_success_rate'])}",
                "",
            ]
        )

    failures = _failure_rows(report["cases"])
    lines.extend(["## 需要分析的检索或端到端样本", ""])
    if not failures:
        lines.append("本次定义的核心判定中没有失败样本。")
    else:
        lines.extend(["| ID | 问题 | 失败原因 |", "| --- | --- | --- |"]) 
        for item in failures:
            lines.append(f"| {item['id']} | {item['question']} | {item['reason']} |")
    lines.append("")
    lines.append("> 本报告只代表当前文档、评测集、模型版本与参数下的结果。")
    lines.append("")
    return "\n".join(lines)


def _failure_rows(cases: list[dict]) -> list[dict]:
    failures: list[dict] = []
    for case in cases:
        reasons: list[str] = []
        if case["expected_behavior"] == "answer":
            if case["metrics"]["fused"]["hit_at_5"] == 0:
                reasons.append("RRF Hit@5 未命中正确页")
            if case["predicted_behavior"] == "refuse":
                reasons.append("可回答问题被检索门控拒答")
        elif (
            case["retrieval_expectation"] == "refuse"
            and case["predicted_behavior"] == "answer"
        ):
            reasons.append("知识库外问题被错误放行")
        generation = case.get("generation")
        if generation is not None and not generation["success"]:
            reasons.append("端到端回答未满足预期")
        if reasons:
            failures.append(
                {
                    "id": case["id"],
                    "question": case["question"].replace("|", "\\|"),
                    "reason": "；".join(dict.fromkeys(reasons)),
                }
            )
    return failures


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"
