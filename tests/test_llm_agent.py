import json
from datetime import date
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest

from utils.llm_agent import (
    DEFAULT_HOST,
    PARAMS,
    AnalystError,
    build_context,
    chat,
    fingerprint,
    load_reference,
    query_data,
    summarize,
    validate_host,
)

TODAY = date(2026, 10, 10)


def args(**overrides):
    result = dict.fromkeys(PARAMS)
    result.update(operation="stats", offset=0)
    result.update(overrides)
    return result


def response(text="Verified response", calls=None, **overrides):
    return httpx.Response(
        200,
        request=httpx.Request("POST", DEFAULT_HOST + "/api/chat"),
        json={
            "done": True,
            "message": {"role": "assistant", "content": text, "tool_calls": calls or []},
            **overrides,
        },
    )


def client_response(text="Verified response"):
    client = Mock()
    client.post.return_value = response(text)
    return client


def test_context_exact_dataset_units_scope_metrics(sample):
    context = build_context(sample, "Fahrenheit", "2026-09-01", "2026-09-30", today=TODAY)
    assert context["scope"] == ["2026-09-01", "2026-09-30"]
    assert context["verified_metrics"]["duration"] == 15.5
    assert context["verified_metrics"]["temperature"] == 176
    assert context["verified_metrics"]["paired_hr_change"] == 20
    assert context["units"]["Sauna Temp"] == "Fahrenheit"
    assert len(context["sessions"]) == 6
    assert context["sessions"][0]["Heart Rate Pre-Sauna"] == 63
    assert context["recommendation_window"] == ["2026-10-10", "2027-04-10"]
    assert context["reference"]["verified"]
    assert all(key in context["sessions"][0] for key in context["fields_available"])


@pytest.mark.parametrize(
    "host",
    [
        "",
        "https://localhost:11434",
        "http://ollama.com",
        "http://localhost@evil.com",
        "http://localhost:11434/api",
        "http://localhost:0",
    ],
)
def test_reject_nonlocal_host_no_api(sample, host):
    client = client_response()
    with pytest.raises(AnalystError, match="OLLAMA_HOST"):
        chat(sample, "Celsius", "Hello", [], host, client=client)
    client.post.assert_not_called()


@pytest.mark.parametrize(
    "exception,message",
    [
        (httpx.ReadTimeout("private payload"), "timed out"),
        (httpx.ConnectError("private payload"), "connect"),
        (
            httpx.HTTPStatusError(
                "private payload",
                request=httpx.Request("POST", DEFAULT_HOST),
                response=httpx.Response(404),
            ),
            "unavailable",
        ),
        (
            httpx.HTTPStatusError(
                "private payload",
                request=httpx.Request("POST", DEFAULT_HOST),
                response=httpx.Response(400),
            ),
            "supports tools",
        ),
    ],
)
def test_safe_mocked_api_failures(sample, exception, message):
    client = client_response()
    client.post.side_effect = exception
    with pytest.raises(AnalystError, match=message) as caught:
        chat(sample, "Celsius", "Explain", [], DEFAULT_HOST, client=client)
    assert "private payload" not in str(caught.value)
    assert len(sample) == 17


def test_request_full_data_history_local(sample):
    client = client_response()
    history = [
        {"role": "user", "content": "My earlier question"},
        {"role": "assistant", "content": "Earlier response"},
    ]
    assert (
        chat(sample, "Celsius", "Explain", history, DEFAULT_HOST, client=client, today=TODAY)
        == "Verified response"
    )
    request = client.post.call_args.kwargs["json"]
    assert client.post.call_args.args[0] == DEFAULT_HOST + "/api/chat"
    assert request["stream"] is False
    assert request["messages"][2:4] == history
    context = json.loads(request["messages"][1]["content"].split(": ", 1)[1])
    assert context["verified_metrics"]["session_count"] == 17
    assert context["scope"] == ["2025-12-28", "2026-10-05"]
    assert request["tools"][0]["function"]["name"] == "query_data"


def test_month_and_no_data_and_ambiguity(sample):
    result = query_data(sample, "Celsius", args(operation="month", month=9))
    assert result["range"] == ["2026-09-01", "2026-09-30"]
    assert result["verified_metrics"]["total_minutes"] == 93
    assert query_data(sample, "Celsius", args(operation="month", month=2))["no_data"]
    another = sample.copy()
    another.date = another.date - pd.DateOffset(years=1)
    combined = pd.concat([sample, another], ignore_index=True).sort_values("date")
    assert "clarification_required" in query_data(
        combined, "Celsius", args(operation="month", month=9)
    )
    client = client_response()
    assert "Which year" in chat(
        combined, "Celsius", "Summarize September", [], DEFAULT_HOST, client=client
    )
    client.post.assert_not_called()


def test_tool_roundtrip_month_verified(sample):
    client = client_response()
    call = {"function": {"name": "query_data", "arguments": {"operation": "month", "month": 9}}}
    client.post.side_effect = [
        response("", [call]),
        response("September 2026 had 6 sessions and 93 minutes."),
    ]
    assert "93 minutes" in chat(
        sample, "Celsius", "Summarize September", [], DEFAULT_HOST, client=client
    )
    output = client.post.call_args.kwargs["json"]["messages"][-1]
    assert json.loads(output["content"])["verified_metrics"]["total_minutes"] == 93


def test_tool_comparison_and_reject_arbitrary_operations(sample):
    comparison = query_data(
        sample,
        "Celsius",
        args(
            operation="compare",
            start="2026-08-01",
            end="2026-08-31",
            other_start="2026-09-01",
            other_end="2026-09-30",
        ),
    )
    assert comparison["b_minus_a"]["pre_hr"] == -0.5
    for bad in (
        args(operation="exec"),
        args(offset=-1),
        args(start="September"),
        args(operation="aggregate", grouping="Daily"),
        args(operation="month", month=13),
    ):
        with pytest.raises(ValueError):
            query_data(sample, "Celsius", bad)


def test_large_context_explicit_paginated_access(sample):
    large = pd.concat([sample] * 20, ignore_index=True).sort_values("date")
    context = build_context(large, today=TODAY)
    assert context["sessions"]["rows"] == 340
    assert "every row" in context["sessions"]["access"]
    page = query_data(large, "Fahrenheit", args(operation="sessions", offset=300))
    assert page["total"] == 340 and len(page["records"]) == 40
    assert page["next_offset"] is None
    assert all("Sauna Temp" in r for r in page["records"])


def test_summary_schema_and_invalidation(sample):
    client = client_response(
        json.dumps(
            {
                "sentences": [
                    "Your readings describe your recorded sessions.",
                    "Track timing consistently for the next six months.",
                    "History cannot predict health outcomes.",
                ]
            }
        )
    )
    assert "six months" in summarize(
        sample, "Celsius", "2026-09-01", "2026-09-30", DEFAULT_HOST, today=TODAY, client=client
    )
    assert client.post.call_args.kwargs["json"]["format"]["type"] == "object"
    original = fingerprint(sample, "Celsius", "2026-09-01", "2026-09-30", "a", {}, TODAY)
    assert original != fingerprint(sample, "Fahrenheit", "2026-09-01", "2026-09-30", "a", {}, TODAY)
    assert original != fingerprint(sample, "Celsius", "2026-09-01", "2026-09-30", "b", {}, TODAY)
    assert original != fingerprint(
        sample.iloc[1:], "Celsius", "2026-09-01", "2026-09-30", "a", {}, TODAY
    )


def test_reference_fallback_and_summary_invalid(sample, tmp_path):
    assert not load_reference(tmp_path / "missing.json")["verified"]
    client = client_response("malformed")
    with pytest.raises(AnalystError, match="invalid summary"):
        summarize(sample, "Celsius", "2026-09-01", "2026-09-30", DEFAULT_HOST, client=client)


def test_corrupt_reference_falls_back(tmp_path):
    for text in (
        "[]",
        "not-json",
        '{"verified": true, "claims": ["bad"], "source_url": "https://www.foundmyfitness.com/topics/sauna"}',
    ):
        path = tmp_path / "reference.json"
        path.write_text(text)
        assert not load_reference(path)["verified"]


def test_summary_rejects_multiple_sentences_per_item(sample):
    client = client_response(
        json.dumps(
            {
                "sentences": [
                    "One sentence. Another sentence.",
                    "Third sentence.",
                    "Fourth sentence.",
                ]
            }
        )
    )
    with pytest.raises(AnalystError, match="invalid summary"):
        summarize(sample, "Celsius", "2026-09-01", "2026-09-30", DEFAULT_HOST, client=client)


@pytest.mark.parametrize(
    "host", ["http://127.0.0.1:11434", "http://[::1]:11434", DEFAULT_HOST + "/"]
)
def test_local_hosts(host):
    assert validate_host(host) == host.rstrip("/")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"done": True},
        {"done": False, "message": {"role": "assistant"}},
        {
            "done": True,
            "done_reason": "length",
            "message": {"role": "assistant", "content": "partial"},
        },
    ],
)
def test_bad_or_incomplete_response(sample, payload):
    client = client_response()
    client.post.return_value = httpx.Response(
        200, request=httpx.Request("POST", DEFAULT_HOST), json=payload
    )
    with pytest.raises(AnalystError):
        chat(sample, "Celsius", "Explain", [], client=client)


def test_unknown_tool_is_rejected(sample):
    client = client_response()
    client.post.side_effect = [
        response("", [{"function": {"name": "exec", "arguments": {}}}]),
        response(),
    ]
    chat(sample, "Celsius", "Explain", [], client=client)
    tool = client.post.call_args.kwargs["json"]["messages"][-1]
    assert "Only query_data is allowed" in json.loads(tool["content"])["query_error"]


def test_default_model_and_adaptive_context(sample):
    from utils.llm_agent import generation_options

    client = client_response()
    chat(sample, "Celsius", "Explain", [], client=client)
    payload = client.post.call_args.kwargs["json"]
    assert "think" not in payload
    assert 4096 <= payload["options"]["num_ctx"] < 32768
    small = generation_options([{"role": "user", "content": "Hello"}], 1024)
    large = generation_options([{"role": "user", "content": "a" * 50000}], 2048)
    assert small["num_ctx"] == 4096
    assert large["num_ctx"] > small["num_ctx"]


def test_summary_invalidates_on_host_change(sample):
    original = fingerprint(sample, "Celsius", None, None, "model", {}, TODAY)
    assert original != fingerprint(
        sample, "Celsius", None, None, "model", {}, TODAY, "http://localhost:12345"
    )


def test_native_http_transport_stays_local(sample, monkeypatch):
    import utils.llm_agent as agent

    actual_client = httpx.Client
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url == DEFAULT_HOST + "/api/chat"
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200, json={"done": True, "message": {"role": "assistant", "content": "Local response"}}
        )

    def factory(**kwargs):
        assert kwargs == {"timeout": 120.0, "trust_env": False, "follow_redirects": False}
        return actual_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(agent.httpx, "Client", factory)
    assert chat(sample, "Celsius", "Explain", []) == "Local response"
    assert len(requests) == 1
