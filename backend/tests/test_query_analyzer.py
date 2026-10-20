from app.retrieval.query_analyzer import QueryAnalyzer


def test_query_analyzer_detects_model_and_intent():
    plan = QueryAnalyzer().analyze("G20 无法联网怎么办？")

    assert plan.model == "G20"
    assert plan.intent == "配网与 App"
    assert plan.variants == ("无法联网怎么办？",)


def test_query_analyzer_normalizes_explicit_model_alias():
    plan = QueryAnalyzer().analyze("如何清理尘盒？", "g10spro")

    assert plan.model == "G10S Pro"


def test_query_analyzer_treats_all_as_no_model_filter():
    plan = QueryAnalyzer().analyze("如何清理尘盒？", "all")

    assert plan.model is None


def test_query_analyzer_removes_model_alias_because_metadata_filter_handles_it():
    plan = QueryAnalyzer().analyze("石头G10SPro 如何清理尘盒？")

    assert plan.model == "G10S Pro"
    assert plan.variants == ("如何清理尘盒？",)

