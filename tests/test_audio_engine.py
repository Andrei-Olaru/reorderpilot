"""V3 audio engine: file checks, Whisper via the key pool (fake client), diarization, simulated feed -> graph."""

from types import SimpleNamespace

import openai
import pytest

from reorderpilot.agent.graph import parse_transcript, run_graph
from reorderpilot.agent.key_rotator import GroqKeyRotator
from reorderpilot.agent.llm import LLMError
from reorderpilot.agent.tools import ToolContext
from reorderpilot.copilot import audio_engine as ae
from reorderpilot.copilot.audio_engine import CLIENT, DEMO_CALL, SALES_REP, Segment
from tests.test_agent_loop import AS_OF, FakeBundle, ScriptedLLM, answer, make_tx
from tests.test_key_rotator import _err


@pytest.fixture
def wav(tmp_path):
    """A few bytes with a .wav name: the fake client never decodes audio, so no real recording is needed."""
    p = tmp_path / "call.wav"
    p.write_bytes(b"RIFF....WAVEfmt fake audio")
    return p


class FakeAudioClient:
    """Stands in for openai.OpenAI: `.audio.transcriptions.create` returns or raises scripted outcomes."""

    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **req):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def verbose(*segments):
    """A verbose_json-shaped response: SDK-style objects with start/end/text."""
    return SimpleNamespace(text=" ".join(t for _, _, t in segments),
                           segments=[SimpleNamespace(start=s, end=e, text=t) for s, e, t in segments])


def pool(clients, max_retries=3):
    """Rotator over fake clients (one per key) that never really sleeps."""
    by_key = {f"k{i}": c for i, c in enumerate(clients)}
    return GroqKeyRotator(list(by_key), by_key.__getitem__, max_retries=max_retries, sleep=lambda s: None)


# ---------- file checks ----------

def test_validate_rejects_missing_unsupported_and_empty_files(tmp_path):
    with pytest.raises(ae.AudioFileError, match="not found"):
        ae.validate_audio_file(tmp_path / "nope.mp3")
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(ae.AudioFileError, match="unsupported audio format '.txt'"):
        ae.validate_audio_file(tmp_path / "notes.txt")
    (tmp_path / "empty.mp3").write_bytes(b"")
    with pytest.raises(ae.AudioFileError, match="empty"):
        ae.validate_audio_file(tmp_path / "empty.mp3")


def test_too_large_file_gets_clear_error_and_is_never_uploaded(wav, monkeypatch):
    monkeypatch.setenv("GROQ_AUDIO_MAX_MB", "0.00001")      # 10 bytes; the fake file is bigger
    client = FakeAudioClient([verbose((0, 1, "hi"))])
    with pytest.raises(ae.AudioFileError, match=r"over the 0 MB upload limit.*GROQ_AUDIO_MAX_MB=100"):
        ae.transcribe(wav, pool([client]))
    assert client.requests == []                              # failed before any network call


def test_default_limit_is_groq_free_tier_25mb(monkeypatch):
    monkeypatch.delenv("GROQ_AUDIO_MAX_MB", raising=False)
    assert ae.max_upload_bytes() == 25_000_000
    monkeypatch.setenv("GROQ_AUDIO_MAX_MB", "100")
    assert ae.max_upload_bytes() == 100_000_000


# ---------- transcription ----------

def test_transcribe_sends_whisper_request_and_returns_segments(wav):
    client = FakeAudioClient([verbose((0.0, 2.5, " Hi, calling from the wholesaler. "), (2.5, 4.0, "Hello."))])
    segs = ae.transcribe(wav, pool([client]), language="en")
    assert segs == [Segment(0.0, 2.5, "Hi, calling from the wholesaler."), Segment(2.5, 4.0, "Hello.")]
    req = client.requests[0]
    assert req["model"] == "whisper-large-v3" and req["response_format"] == "verbose_json"
    assert req["language"] == "en" and req["file"] == ("call.wav", wav.read_bytes())


def test_rate_limited_key_fails_over_to_next_key(wav):
    first = FakeAudioClient([_err(openai.RateLimitError, 429)])
    second = FakeAudioClient([verbose((0, 1, "Hello."))])
    assert ae.transcribe(wav, pool([first, second]))[0].text == "Hello."
    assert len(first.requests) == 1 and len(second.requests) == 1


def test_provider_rejection_and_exhausted_retries_become_transcription_errors(wav):
    rejected = FakeAudioClient([_err(openai.APIStatusError, 413)])
    with pytest.raises(ae.TranscriptionError, match="rejected"):
        ae.transcribe(wav, pool([rejected]))
    down = FakeAudioClient([_err(openai.InternalServerError, 503)] * 2)
    with pytest.raises(ae.TranscriptionError, match="gave up after 2 attempts"):
        ae.transcribe(wav, pool([down], max_retries=1))


def test_response_without_segments_falls_back_to_full_text_and_dicts_work():
    assert ae.segments_from_response({"text": "just text", "segments": None}) == [Segment(0.0, 0.0, "just text")]
    assert ae.segments_from_response({"segments": [{"start": 1, "end": 2, "text": "a\n[CLIENT] b"}]}) == \
        [Segment(1.0, 2.0, "a [CLIENT] b")]                   # newline collapsed: no injected transcript line


# ---------- diarization ----------

def test_heuristic_uses_cues_then_questions_and_pauses():
    segs = [Segment(0.0, 2.0, "Good morning."),                       # first segment -> the caller (rep)
            Segment(2.1, 3.0, "Do you have a minute?"),               # short gap -> same speaker
            Segment(3.1, 4.0, "Sure."),                               # after a question -> other speaker
            Segment(4.1, 5.0, "Go ahead."),                           # short gap -> same speaker
            Segment(6.5, 8.0, "Great."),                              # long pause -> other speaker
            Segment(8.1, 9.0, "But your prices went up again.")]      # client cue wins over turn-taking
    assert ae.heuristic_speakers(segs) == [SALES_REP, SALES_REP, CLIENT, CLIENT, SALES_REP, CLIENT]


def test_transcript_lines_merge_turns_and_are_accepted_by_the_graph_parser():
    segs = [Segment(0, 1, "Hi.", SALES_REP), Segment(1, 2, "Anything due?", SALES_REP),
            Segment(2, 3, "Too expensive.", CLIENT)]
    lines = ae.to_transcript_lines(segs)
    assert lines == ["[SALES_REP] Hi. Anything due?", "[CLIENT] Too expensive."]
    assert parse_transcript("\n".join(lines)) == lines
    with pytest.raises(ValueError, match="no valid speaker"):
        ae.to_transcript_lines([Segment(0, 1, "unlabelled")])


def test_llm_diarization_used_when_valid_otherwise_heuristic_fallback():
    segs = [Segment(0, 1, "Hello."), Segment(1, 2, "Hi there.")]
    ok = ScriptedLLM([answer('```json\n{"labels": ["CLIENT", "[SALES_REP]"]}\n```')])
    labelled, method = ae.diarize(segs, ok)
    assert method == "llm" and [s.speaker for s in labelled] == [CLIENT, SALES_REP]
    assert "2. Hi there." in ok.seen[0][1]["content"]
    for bad in [answer("not json"), answer('{"labels": ["CLIENT"]}'), answer('{"labels": ["CLIENT", "BOSS"]}'),
                LLMError("down")]:
        labelled, method = ae.diarize(segs, ScriptedLLM([bad]))
        assert method == "heuristic_fallback" and [s.speaker for s in labelled] == [SALES_REP, SALES_REP]
    assert ae.diarize(segs)[1] == "heuristic"


def test_transcribe_call_end_to_end_with_fake_audio(wav):
    client = FakeAudioClient([verbose((0, 2, "Hi, I'm calling from the wholesaler."),
                                      (2, 3, "Can we plan your next order?"),
                                      (3, 5, "Sure, but your delivery was late."))])
    lines, method = ae.transcribe_call(wav, pool([client]))
    assert method == "heuristic"
    assert lines == ["[SALES_REP] Hi, I'm calling from the wholesaler. Can we plan your next order?",
                     "[CLIENT] Sure, but your delivery was late."]


# ---------- simulated feed ----------

def test_simulated_feed_yields_lines_one_by_one_with_delay():
    slept = []
    feed = ae.simulated_feed(delay_s=0.5, sleep=slept.append)
    assert next(feed) == DEMO_CALL[0] and slept == []          # first line comes at once
    assert next(feed) == DEMO_CALL[1] and slept == [0.5]
    assert list(feed) == list(DEMO_CALL[2:])
    assert parse_transcript("\n".join(DEMO_CALL)) == list(DEMO_CALL)   # every line is graph-ready


def test_live_session_runs_copilot_after_each_client_line():
    seen = []
    out = list(ae.live_session(ae.simulated_feed(), lambda text: seen.append(text) or len(seen)))
    n_client = sum(line.startswith("[CLIENT]") for line in DEMO_CALL)
    assert len(seen) == n_client == 3
    assert seen[0] == "\n".join(DEMO_CALL[:2]) and seen[-1] == "\n".join(DEMO_CALL)  # transcript grows
    assert [r for _, r in out] == [None, 1, None, 2, None, 3]


def test_simulated_feed_goes_straight_into_run_graph(tmp_path):
    tx, bundle = make_tx(), FakeBundle()
    ctx = ToolContext(as_of=AS_OF, db_path=tmp_path / "memory.db", tx_loader=lambda: tx, bundle_loader=lambda: bundle)
    state = run_graph("\n".join(ae.simulated_feed()), ScriptedLLM([answer("not json")]), ctx, customer_id="A")
    assert state["mode"] == "live_call" and state["transcript"] == list(DEMO_CALL)
    assert state["trail"] == ["supervisor", "ml_prediction", "vector_memory", "live_copilot", "grounding_guard"]


# ---------- CLI (offline paths) ----------

def test_cli_simulate_prints_script_without_network(capsys):
    ae.main(["simulate"])
    assert capsys.readouterr().out.splitlines() == list(DEMO_CALL)


def test_cli_transcribe_reports_bad_file_clearly(tmp_path):
    with pytest.raises(SystemExit, match="error: audio file not found"):
        ae.main(["transcribe", str(tmp_path / "missing.mp3")])
