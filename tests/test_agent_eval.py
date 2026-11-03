"""Eval helpers: hallucinated-product detection and the suppressed-vs-recommended classifier."""

from reorderpilot.agent.evaluation import classify_mention, hallucinated_products, mentioned_products

CAT = {"22629": "SPACEBOY LUNCH BOX", "22630": "DOLLY GIRL LUNCH BOX", "85099B": "JUMBO BAG RED RETROSPOT",
       "14646": "SOME PRODUCT WITH CODE LIKE A CUSTOMER ID"}


def test_mentions_found_by_code_or_description_and_customer_ids_ignored():
    text = "Customer 14646: offer Dolly Girl Lunch Box and 85099B."
    assert mentioned_products(text, CAT, ignore={"14646"}) == {"22630", "85099B"}


def test_hallucination_is_a_product_no_tool_returned():
    outputs = [{"name": "predict_reorder", "result": {"customer_id": "14646",
                                                      "recommendations": [{"stock_code": "22630"}]}}]
    answer = "For 14646: DOLLY GIRL LUNCH BOX (22630) and JUMBO BAG RED RETROSPOT."
    assert hallucinated_products(answer, outputs, CAT, "Brief me on 14646") == {"85099B"}


def test_classify_mention():
    assert classify_mention("Offer SPACEBOY LUNCH BOX (73%)", "22629", "SPACEBOY LUNCH BOX") == "recommended"
    assert classify_mention("22629 is suppressed (overstocked)", "22629", "SPACEBOY LUNCH BOX") == \
        "mentioned_suppressed"
    assert classify_mention("Offer the dolly girl box", "22629", "SPACEBOY LUNCH BOX") == "absent"


def test_unicode_dashes_from_llm_still_match():
    cat = {"85123A": "WHITE HANGING HEART T-LIGHT HOLDER"}
    answer = "1. **WHITE HANGING HEART T\u2011LIGHT HOLDER** \u2013 12.5% probability"
    assert classify_mention(answer, "85123A", cat["85123A"]) == "recommended"
    assert mentioned_products(answer, cat) == {"85123A"}


def test_shorter_description_inside_longer_one_is_not_a_second_product():
    cat = {"85123A": "WHITE HANGING HEART T-LIGHT HOLDER", "21814": "HEART T-LIGHT HOLDER"}
    assert mentioned_products("Offer the WHITE HANGING HEART T-LIGHT HOLDER", cat) == {"85123A"}
