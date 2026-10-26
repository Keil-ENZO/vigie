import json

import httpx

from vigie import analyst


class Reply:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_openai_backend_sends_image_and_schema(monkeypatch):
    sent = {}

    def post(url, json, headers, timeout):
        sent.update(url=url, body=json, headers=headers)
        return Reply({"choices": [{"message": {"content": '```json\n{"is_vessel": true}\n```'}}]})

    monkeypatch.setattr(analyst, "API", "openai")
    monkeypatch.setattr(analyst, "URL", "http://localhost:1234/v1")
    monkeypatch.setattr(analyst, "KEY", "k")
    monkeypatch.setattr(httpx, "post", post)
    raw = analyst._chat("décris", ["QUJD"], analyst.VESSEL_SCHEMA)

    assert sent["url"] == "http://localhost:1234/v1/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer k"}
    parts = sent["body"]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert sent["body"]["response_format"]["json_schema"]["schema"] is analyst.VESSEL_SCHEMA
    assert analyst._json(raw) == {"is_vessel": True}


def test_ollama_backend_keeps_native_api(monkeypatch):
    sent = {}

    def post(url, json, headers, timeout):
        sent.update(url=url, body=json, headers=headers)
        return Reply({"message": {"content": '{"ok": 1}'}})

    monkeypatch.setattr(analyst, "API", "ollama")
    monkeypatch.setattr(analyst, "URL", "http://localhost:11434")
    monkeypatch.setattr(analyst, "KEY", None)
    monkeypatch.setattr(httpx, "post", post)
    raw = analyst._chat("décris", ["QUJD"], analyst.VESSEL_SCHEMA)

    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["headers"] == {}
    assert sent["body"]["messages"][0]["images"] == ["QUJD"]
    assert sent["body"]["format"] is analyst.VESSEL_SCHEMA
    assert json.loads(raw) == {"ok": 1}


def test_available_accepts_latest_tag(monkeypatch):
    monkeypatch.setattr(analyst, "MODEL", "qwen2.5vl")
    monkeypatch.setattr(analyst, "models", lambda: ["qwen2.5vl:latest"])
    assert analyst.available()
    monkeypatch.setattr(analyst, "models", list)
    assert not analyst.available()
