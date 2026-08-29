from types import SimpleNamespace
from unittest.mock import Mock, patch

from forge.smoke import main


def test_smoke_entry_sends_one_text_request(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    model_client = Mock()
    model_client.create_response.return_value = SimpleNamespace(
        model="gpt-5.6-sol",
        response_id="resp_smoke",
        status="completed",
        output_text="Forge smoke test successful.",
    )

    with patch("forge.smoke.OpenAIResponsesClient", return_value=model_client):
        exit_code = main(["Say hello"])

    assert exit_code == 0
    request = model_client.create_response.call_args.args[0]
    assert request.input == "Say hello"
    assert request.tools == ()
    output = capsys.readouterr().out
    assert "gpt-5.6-sol" in output
    assert "Forge smoke test successful." in output


def test_smoke_entry_fails_cleanly_without_api_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = main([])

    assert exit_code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
