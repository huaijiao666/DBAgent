from calculator import multiply


def test_multiply_returns_the_product() -> None:
    assert multiply(6, 7) == 42
