"""Renders the Streamlit UI headlessly against a real (in-process) backend with fake models and LLM."""
import time
import httpx
import pytest
import socket
import uvicorn
import threading
from ragchat.api import create_app
from ragchat.config import Settings
from conftest import ROOT, HashEmbedder, OverlapReranker


@pytest.fixture(scope="module")
def backend(tmp_path_factory, monkeypatch_module):
    def stream(msgs):
        yield "The battery lasts 14 hours [1]."

    def complete(msgs, **kw):
        return "battery life"

    app = create_app(Settings("http://llm.invalid/v1", "k", "m", tmp_path_factory.mktemp("ui") / "ui.db", 5),
                     stream_fn=stream, complete_fn=complete, embedder=HashEmbedder(),
                     reranker=OverlapReranker(), use_local_models=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{url}/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    monkeypatch_module.setenv("BACKEND_URL", url)
    httpx.post(f"{url}/documents", files=[("files", ("manual.txt", b"The Zephyr-9 battery lasts 14 hours."))])
    yield url
    server.should_exit = True


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_ui_renders_all_tabs_and_chats(backend):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Chat", "Library", "Insights", "Evaluation"]
    assert dict((m.label, m.value) for m in at.metric)["Documents"] == "1"

    at.chat_input[0].set_value("How long does the battery last?").run()
    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == "The battery lasts 14 hours [1]."

    at.button(key="down1").click().run()
    assert not at.exception
    assert httpx.get(f"{backend}/analytics").json()["totals"]["thumbs_down"] == 1


def test_ui_retrieval_evaluation_button(backend):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    next(b for b in at.button if b.label == "Run retrieval evaluation").click().run()
    assert not at.exception
    assert at.session_state.ret_eval["items"] == 24
