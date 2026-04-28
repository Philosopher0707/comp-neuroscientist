"""Tests for config module."""

from comp_neuroscientist.config import Config, _safe_int_env


def test_default_model():
    c = Config()
    assert c.model == "deepseek-v4-flash:cloud"


def test_default_local_model(monkeypatch):
    monkeypatch.setenv("CN_LOCAL", "1")
    c = Config()
    assert c.model == "llama3.1"


def test_local_flag_overrides_env_model(monkeypatch):
    """CN_MODEL takes precedence over CN_LOCAL."""
    monkeypatch.setenv("CN_LOCAL", "1")
    monkeypatch.setenv("CN_MODEL", "glm-5.1:cloud")
    c = Config()
    assert c.model == "glm-5.1:cloud"


def test_local_default_false():
    c = Config()
    assert not c.local


def test_local_env_true(monkeypatch):
    monkeypatch.setenv("CN_LOCAL", "true")
    assert Config.from_env().local


def test_local_env_1(monkeypatch):
    monkeypatch.setenv("CN_LOCAL", "1")
    assert Config.from_env().local


def test_local_env_yes(monkeypatch):
    monkeypatch.setenv("CN_LOCAL", "yes")
    assert Config.from_env().local

def test_env_override(monkeypatch):
    monkeypatch.setenv("CN_MODEL", "glm-5.1:cloud")
    c = Config()
    assert c.model == "glm-5.1:cloud"


def test_max_turns_default():
    c = Config()
    assert c.max_turns == 30


def test_max_turns_env(monkeypatch):
    monkeypatch.setenv("CN_MAX_TURNS", "50")
    c = Config()
    assert c.max_turns == 50


def test_output_dir_default():
    c = Config()
    assert c.output_dir == "results"


def test_output_dir_env(monkeypatch):
    monkeypatch.setenv("CN_OUTPUT_DIR", "/tmp/neuro")
    c = Config()
    assert c.output_dir == "/tmp/neuro"


def test_plot_dir_property():
    c = Config(output_dir="test_out")
    assert c.plots_dir == "test_out/plots"


def test_safe_int_env_valid(monkeypatch):
    monkeypatch.setenv("TEST_VAL", "42")
    assert _safe_int_env("TEST_VAL", 10) == 42


def test_safe_int_env_empty(monkeypatch):
    monkeypatch.setenv("TEST_VAL", "")
    assert _safe_int_env("TEST_VAL", 10) == 10


def test_safe_int_env_invalid(monkeypatch):
    monkeypatch.setenv("TEST_VAL", "not_a_number")
    assert _safe_int_env("TEST_VAL", 10) == 10


def test_from_env(monkeypatch):
    monkeypatch.setenv("CN_MODEL", "glm-5.1:cloud")
    monkeypatch.setenv("CN_MAX_TURNS", "50")
    c = Config.from_env()
    assert c.model == "glm-5.1:cloud"
    assert c.max_turns == 50
