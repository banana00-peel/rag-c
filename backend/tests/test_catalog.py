from app.ingestion.catalog import detect_brand, detect_model, normalize_model


def test_detects_roborock_model_from_filename_and_text():
    assert detect_brand("石头G20说明书.pdf", "") == "石头"
    assert detect_model("manual.pdf", "适用于石头 G20 扫拖机器人") == "G20"


def test_normalizes_known_model_aliases():
    assert normalize_model("g10spro") == "G10S Pro"
    assert normalize_model("S5") == "S5"

