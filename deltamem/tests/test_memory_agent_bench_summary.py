from __future__ import annotations

from typing import cast

from deltamem.eval.benchmark_compare import summarize_memory_agent_bench


def _record(
    split: str,
    source: str,
    *,
    correct: bool,
    f1: float | None = None,
    recsys_recall_at_5: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "split": split,
        "source": source,
        "correct": correct,
        "prediction": "stub",
        "answer_aliases": ["stub"],
    }
    if f1 is not None:
        record["f1"] = f1
    if recsys_recall_at_5 is not None:
        record["recsys_recall@5"] = recsys_recall_at_5
    return record


def test_memory_agent_bench_summary_matches_weighted_qwen3_8b_style_aggregation() -> None:
    records = [
        _record("Accurate_Retrieval", "ruler_qa1_197K", correct=True),
        _record("Accurate_Retrieval", "eventqa_131072", correct=False),
        _record("Accurate_Retrieval", "eventqa_full", correct=True),
        _record("Accurate_Retrieval", "longmemeval_s*", correct=False, f1=0.9),
        _record("Test_Time_Learning", "recsys_redial_full", correct=False, recsys_recall_at_5=0.4),
        _record("Test_Time_Learning", "recsys_redial_full", correct=False, recsys_recall_at_5=0.0),
        _record("Long_Range_Understanding", "infbench_sum_eng_shots2", correct=False, f1=0.2),
        _record("Conflict_Resolution", "factconsolidation_sh_full", correct=True),
        _record("Conflict_Resolution", "factconsolidation_mh_full", correct=False),
        _record("Conflict_Resolution", "factconsolidation_mh_full", correct=True),
    ]

    summary = cast(dict[str, object], summarize_memory_agent_bench(records))
    categories = cast(dict[str, dict[str, object]], summary["categories"])
    category_overall = cast(dict[str, float], summary["category_overall"])
    category_sample_weights = cast(dict[str, int], summary["category_sample_weights"])
    dataset_scores = cast(dict[str, dict[str, object]], summary["dataset_scores"])

    assert summary["primary_metric"] == "sample_weighted_category_overall"
    assert summary["overall"] == 0.46
    assert category_overall["Accurate Retrieval"] == 0.5
    assert category_overall["Test-time Learning"] == 0.2
    assert category_overall["Long Range Understanding"] == 0.2
    assert category_overall["Selective Forgetting"] == 0.6667
    assert category_sample_weights["Accurate Retrieval"] == 4
    assert category_sample_weights["Test-time Learning"] == 2
    assert category_sample_weights["Long Range Understanding"] == 1
    assert category_sample_weights["Selective Forgetting"] == 3

    accurate = cast(dict[str, object], categories["Accurate Retrieval"])
    accurate_datasets = cast(dict[str, dict[str, object]], accurate["datasets"])
    assert accurate["overall"] == 0.5
    assert accurate_datasets["SH-Doc QA"]["score"] == 1.0
    assert accurate_datasets["EventQA"]["metric_key"] == "accuracy"
    assert accurate_datasets["EventQA"]["score"] == 0.5
    assert accurate_datasets["LongMemEval (S*)"]["metric_key"] == "accuracy"
    assert accurate_datasets["LongMemEval (S*)"]["score"] == 0.0

    ttl = cast(dict[str, object], categories["Test-time Learning"])
    ttl_datasets = cast(dict[str, dict[str, object]], ttl["datasets"])
    assert ttl_datasets["Movie Recommendation"]["metric_key"] == "recsys_recall@5"
    assert ttl_datasets["Movie Recommendation"]["score"] == 0.2

    lru = cast(dict[str, object], categories["Long Range Understanding"])
    lru_datasets = cast(dict[str, dict[str, object]], lru["datasets"])
    assert lru_datasets["InfinityBench-Sum"]["metric_key"] == "f1"
    assert lru_datasets["InfinityBench-Sum"]["score"] == 0.2

    selective = cast(dict[str, object], categories["Selective Forgetting"])
    selective_datasets = cast(dict[str, dict[str, object]], selective["datasets"])
    assert selective_datasets["FactConsolidation-SH"]["score"] == 1.0
    assert selective_datasets["FactConsolidation-SH"]["num_samples"] == 1
    assert selective_datasets["FactConsolidation-MH"]["score"] == 0.5
    assert selective_datasets["FactConsolidation-MH"]["num_samples"] == 2
    assert dataset_scores["FactConsolidation-MH"]["category"] == "Selective Forgetting"
