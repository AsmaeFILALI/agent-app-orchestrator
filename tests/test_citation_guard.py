"""Tests for the streaming citation guard (src/util/citation_guard.py)."""

import random

import pytest

from util.citation_guard import (
    MAX_LINK_CHARS,
    MODE_ENFORCE,
    MODE_LOG_ONLY,
    MODE_OFF,
    CitationGuard,
    canonical_url,
    extract_link_destinations,
    resolve_mode,
)

ADO = (
    "https://dev.azure.com/org/proj/_git/repo?path=%2Fdocs%2F07-dns.md"
    "&version=GC9b8d91ea&_a=contents"
)
ADO_2 = "https://dev.azure.com/org/proj/_git/repo?path=%2Fdocs%2F12-residency.md&version=GC9b8d91ea&_a=contents"
FAKE = "https://docs.microsoft.com/azure/ai-services/standards/model-deployment-types"


def guard(allowed=(ADO, ADO_2), mode=MODE_ENFORCE):
    return CitationGuard(lambda: list(allowed), mode=mode)


def run(text, *, allowed=(ADO, ADO_2), mode=MODE_ENFORCE, size=None):
    """Stream ``text`` in fixed-size fragments (or whole) and join the output."""
    g = guard(allowed, mode)
    if size is None:
        return g.feed(text) + g.flush(), g
    out = "".join(g.feed(text[i : i + size]) for i in range(0, len(text), size))
    return out + g.flush(), g


# -- retrieved links ---------------------------------------------------------


def test_retrieved_link_is_unchanged():
    text = f"See [DNS guide]({ADO}) for details."
    out, g = run(text)
    assert out == text
    assert g.stats.altered == 0 and g.stats.links_seen == 1


def test_unretrieved_link_is_replaced_by_its_text():
    out, g = run(f"Read [Allowed deployment types]({FAKE}) first.")
    assert out == "Read Allowed deployment types first."
    assert FAKE not in out
    assert g.stats.unretrieved == 1


def test_the_reported_failure_keeps_text_and_removes_invented_urls():
    text = (
        "Use a regional deployment. "
        f"[Azure AI Services Standards > Model deployment types > Allowed deployment types]({FAKE}) "
        f"• [Azure AI Services Standards > Model availability]({FAKE}#availability)"
    )
    out, g = run(text)
    assert "docs.microsoft.com" not in out
    assert "Allowed deployment types" in out and "Model availability" in out
    assert g.stats.unretrieved == 2


def test_fragment_on_a_retrieved_link_is_allowed():
    out, g = run(f"See [DNS]({ADO}#hybrid-name-resolution).")
    assert out == f"See [DNS]({ADO}#hybrid-name-resolution)."
    assert g.stats.altered == 0


def test_host_case_is_normalised_but_path_is_not():
    assert canonical_url("HTTPS://Dev.Azure.com/Org/x?y=1#frag") == "https://dev.azure.com/Org/x?y=1"
    out, _ = run("See [DNS](https://DEV.azure.com/org/proj/_git/repo?path=%2Fdocs%2F07-dns.md&version=GC9b8d91ea&_a=contents).")
    assert "[DNS](" in out
    out, g = run(f"See [DNS]({ADO.replace('07-dns', '07-DNS')}).")
    assert out == "See DNS." and g.stats.unretrieved == 1


@pytest.mark.parametrize(
    "destination",
    ["docs/07-dns.md", "/docs/07-dns.md", "javascript:alert(1)", "mailto:a@b.c", "ftp://x/y", f"{ADO}&extra=1"],
)
def test_anything_but_an_exact_retrieved_link_is_stripped(destination):
    out, g = run(f"See [DNS]({destination}).")
    assert out == "See DNS." and g.stats.unretrieved == 1


def test_in_page_anchor_is_allowed():
    out, g = run("Jump to [the table](#hybrid).")
    assert out == "Jump to [the table](#hybrid)." and g.stats.altered == 0


def test_optional_title_and_angle_brackets_are_understood():
    out, _ = run(f'See [DNS]({ADO} "The DNS guide").')
    assert out == f'See [DNS]({ADO} "The DNS guide").'
    out, _ = run(f"See [DNS](<{ADO}>).")
    assert out == f"See [DNS](<{ADO}>)."
    out, _ = run(f'See [DNS]({FAKE} "x").')
    assert out == "See DNS."


def test_links_the_conversation_already_contains_are_allowed():
    # A follow-up with no retrieval may re-cite a link from an earlier, validated turn.
    history = extract_link_destinations(f"Earlier answer cites [DNS]({ADO}) and [Other]({ADO_2}).")
    out, g = run(f"Again, [DNS]({ADO}).", allowed=history)
    assert out == f"Again, [DNS]({ADO})." and g.stats.altered == 0


# -- repeated citations ------------------------------------------------------


def test_trailing_repeat_citation_is_dropped():
    text = f"Use the inbound endpoint. [DNS]({ADO})\nForward the zone. [DNS]({ADO})\nCheck TTL. [DNS]({ADO})"
    out, g = run(text)
    assert out == f"Use the inbound endpoint. [DNS]({ADO})\nForward the zone. \nCheck TTL. "
    assert g.stats.repeated == 2


def test_repeat_in_a_list_or_sentence_keeps_the_plain_title():
    text = f"According to [DNS]({ADO}), x.\n- [DNS]({ADO})\n1. [DNS]({ADO})\nAnd per [DNS]({ADO}), y."
    out, g = run(text)
    assert out == f"According to [DNS]({ADO}), x.\n- DNS\n1. DNS\nAnd per DNS, y."
    assert g.stats.repeated == 3


def test_different_sources_are_each_cited_once():
    out, g = run(f"A [DNS]({ADO}). B [Residency]({ADO_2}). C [DNS]({ADO}).")
    assert out == f"A [DNS]({ADO}). B [Residency]({ADO_2}). C DNS."
    assert g.stats.repeated == 1


def test_a_stripped_unretrieved_link_does_not_count_as_a_citation():
    out, g = run(f"[Bad]({FAKE}) then [DNS]({ADO}) then [DNS]({ADO}).")
    assert out == f"Bad then [DNS]({ADO}) then DNS."
    assert (g.stats.unretrieved, g.stats.repeated) == (1, 1)


# -- images and unusual markdown --------------------------------------------


def test_unretrieved_image_becomes_alt_text_and_retrieved_image_passes():
    out, g = run(f"![chart]({FAKE}) and ![ok]({ADO})")
    assert out == f"chart and ![ok]({ADO})"
    assert g.stats.unretrieved == 1


@pytest.mark.parametrize(
    "text",
    [
        "Footnote [1] and [2] stay.",
        "Reference style [text][ref] stays.",
        "Array a[0] = 1 and b[i](x) is not valid link? yes it is not retrieved",
        "Escaped \\[not a link\\](https://x.test) text.",
        "A lone [ bracket.",
        "Excited! [ then more",
    ],
)
def test_non_citation_brackets_are_not_swallowed(text):
    out, _ = run(text)
    for fragment in ("[1]", "[text][ref]", "a[0]", "[ bracket", "[ then"):
        if fragment in text:
            assert fragment in out


def test_nested_brackets_and_parentheses_in_a_link_are_handled():
    nested = "https://dev.azure.com/org/p/_git/r?path=%2Fa_(b).md"
    out, _ = run(f"See [Title [v2]]({nested}).", allowed=(nested,))
    assert out == f"See [Title [v2]]({nested})."
    out, _ = run(f"See [Title [v2]]({nested}).", allowed=())
    assert out == "See Title [v2]."


def test_unterminated_link_at_end_of_stream_is_emitted_as_inert_text():
    out, g = run(f"Cut off [DNS]({FAKE}")
    assert out == f"Cut off [DNS]({FAKE}"
    assert ")" not in out[len("Cut off "):]  # no closing paren: not a clickable link
    assert g.stats.altered == 0


def test_a_blank_line_ends_a_pending_bracket():
    out, _ = run(f"see [a\n\nnext paragraph [DNS]({ADO})")
    assert out == f"see [a\n\nnext paragraph [DNS]({ADO})"


# -- streaming behaviour -----------------------------------------------------

STREAM_CASES = [
    f"Use inbound. [DNS]({ADO})\nTwice. [DNS]({ADO}) and [Bad]({FAKE}) and ![img]({FAKE}).",
    f"- [DNS]({ADO})\n- [Residency]({ADO_2})\n- [DNS]({ADO})\n- [X]({FAKE}#frag)",
    "No links at all. Just [1] footnotes! And an exclamation! [ done",
    f"![a](<{ADO}>) ![b]({ADO} \"t\") [c [d]]({FAKE}) \\[e](x) [f]\n\n[g]({ADO_2})",
    "Wide chars: [naïve – café](" + FAKE + ") 漢字 [ok](" + ADO + ")",
]


@pytest.mark.parametrize("text", STREAM_CASES)
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 16, 64])
def test_output_does_not_depend_on_where_the_stream_is_split(text, size):
    whole, whole_guard = run(text)
    split, split_guard = run(text, size=size)
    assert split == whole
    assert split_guard.stats == whole_guard.stats


@pytest.mark.parametrize("text", STREAM_CASES)
def test_random_splits_match_whole_feed(text):
    rng = random.Random(7)
    expected, expected_guard = run(text)
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(1, 12))))
        pieces = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
        g = guard()
        out = "".join(g.feed(p) for p in pieces) + g.flush()
        assert out == expected and g.stats == expected_guard.stats


def test_plain_text_is_not_delayed():
    g = guard()
    assert g.feed("Hello world, ") == "Hello world, "
    assert g.feed("still streaming") == "still streaming"


def test_only_the_open_link_is_held_back_until_it_completes():
    g = guard()
    assert g.feed("Intro [DNS](https://dev.azure.com/org") == "Intro "
    assert g.feed("/proj/_git/repo?path=%2Fdocs%2F07-dns.md&version=GC9b8d91ea&_a=contents) done") == (
        f"[DNS]({ADO}) done"
    )
    assert g.flush() == ""


def test_a_trailing_exclamation_waits_for_the_next_fragment():
    g = guard()
    assert g.feed("Great!") == "Great"
    assert g.feed(f"[x]({FAKE})") == "x"  # the "!" belonged to an image, now alt text
    g = guard()
    assert g.feed("Great!") == "Great"
    assert g.feed(" Next") == "! Next"


def test_an_overlong_link_does_not_stall_the_stream():
    g = guard()
    out = g.feed("[" + "a" * (MAX_LINK_CHARS + 50))
    assert out.startswith("[") and len(out) > MAX_LINK_CHARS


# -- modes -------------------------------------------------------------------


def test_off_passes_everything_through_untouched():
    text = f"[Bad]({FAKE}) [DNS]({ADO}) [DNS]({ADO})"
    out, g = run(text, mode=MODE_OFF, size=3)
    assert out == text and g.stats.links_seen == 0


def test_log_only_reports_but_never_edits():
    text = f"[Bad]({FAKE}) [DNS]({ADO}) [DNS]({ADO})"
    out, g = run(text, mode=MODE_LOG_ONLY, size=3)
    assert out == text
    assert (g.stats.unretrieved, g.stats.repeated) == (1, 1)


@pytest.mark.parametrize("value", [None, "", "  ", "Enforce", "ENFORCE"])
def test_blank_or_mixed_case_mode_means_enforce(value):
    assert resolve_mode(value) == MODE_ENFORCE


def test_unknown_mode_fails_closed(caplog):
    assert resolve_mode("disabled") == MODE_ENFORCE
    assert "Unknown CITATION_GUARD_MODE" in caplog.text


def test_summary_logs_counts_but_never_urls(caplog):
    _, g = run(f"[Bad]({FAKE}) [Bad2]({FAKE}/2)")
    caplog.set_level("WARNING")
    g.log_summary()
    assert "unretrieved=2" in caplog.text
    assert "docs.microsoft.com" not in caplog.text


def test_summary_is_silent_when_nothing_changed(caplog):
    _, g = run(f"[DNS]({ADO})")
    caplog.set_level("WARNING")
    g.log_summary()
    assert caplog.text == ""


def test_allowed_links_are_read_when_each_link_completes():
    retrieved: list[str] = []
    g = CitationGuard(lambda: retrieved)
    first = g.feed(f"See [DNS]({ADO}). ")  # nothing retrieved yet: stripped
    retrieved.append(ADO)
    second = g.feed(f"Again [DNS]({ADO}).")
    assert first == "See DNS. " and second == f"Again [DNS]({ADO})."


# -- history extraction ------------------------------------------------------


def test_extract_link_destinations():
    text = f"A [x]({ADO}) b ![i](<{ADO_2}>) c [n]({FAKE} \"t\") d [1] e [t][r] f [open]("
    assert extract_link_destinations(text) == [ADO, ADO_2, FAKE]
    assert extract_link_destinations("") == [] and extract_link_destinations(None) == []
