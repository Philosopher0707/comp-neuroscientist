"""Tests for subagent definitions."""

from comp_neuroscientist.subagents import ALL_SUBAGENTS


def test_subagents_have_names():
    """All subagents should be registered with a name."""
    assert len(ALL_SUBAGENTS) > 0, "No subagents registered"
    assert "fmri" in ALL_SUBAGENTS
    assert "eeg" in ALL_SUBAGENTS
    assert "ephys" in ALL_SUBAGENTS
    assert "calcium" in ALL_SUBAGENTS
    assert "encoding" in ALL_SUBAGENTS
    assert "simulation" in ALL_SUBAGENTS
    assert "stats" in ALL_SUBAGENTS
    assert "ml" in ALL_SUBAGENTS


def test_subagents_have_descriptions():
    for name, agent in ALL_SUBAGENTS.items():
        assert agent.description, f"{name} has no description"
        assert len(agent.description) > 20, f"{name} description too short"


def test_subagents_have_prompts():
    for name, agent in ALL_SUBAGENTS.items():
        assert agent.prompt, f"{name} has no prompt"
        assert len(agent.prompt) > 100, f"{name} prompt too short"


def test_subagents_have_tools():
    for name, agent in ALL_SUBAGENTS.items():
        assert agent.tools, f"{name} has no tools"
        assert "Bash" in agent.tools, f"{name} missing Bash tool"


def test_subagents_are_unique():
    descriptions = [agent.description[:50] for agent in ALL_SUBAGENTS.values()]
    assert len(descriptions) == len(set(descriptions)), "Duplicate subagent descriptions"


def test_subagent_prompts_mention_domain():
    """Each subagent's prompt should reference its domain."""
    domain_keywords = {
        "fmri": ["bold", "fmri", "brain"],
        "eeg": ["eeg", "meg", "erp", "ica"],
        "ephys": ["spike", "electrophysiology", "psth"],
        "calcium": ["calcium", "suite2p", "deconvolution"],
        "encoding": ["encoding", "rsa", "noise ceiling"],
        "simulation": ["simulation", "lif", "network", "spiking"],
        "stats": ["permutation", "correction", "fdr", "mixed model"],
        "ml": ["pipeline", "cross-validation", "decoder", "classifier"],
    }
    for name, keywords in domain_keywords.items():
        prompt_lower = ALL_SUBAGENTS[name].prompt.lower()
        found = [k for k in keywords if k.lower() in prompt_lower]
        assert found, f"{name}: no domain keywords found in prompt ({keywords})"


def test_subagent_tools_include_read_write():
    for name, agent in ALL_SUBAGENTS.items():
        assert "Read" in agent.tools, f"{name} missing Read"
        assert "Write" in agent.tools, f"{name} missing Write"
