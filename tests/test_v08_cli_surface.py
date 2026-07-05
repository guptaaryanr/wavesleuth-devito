from wavesleuth_devito.challenge import SUPPORTED_CHALLENGES, make_challenge_world
from wavesleuth_devito.cli import build_parser


def test_v08_cli_and_challenge_surface() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "active-demo" in help_text
    assert "mask-cell-easy" in SUPPORTED_CHALLENGES
    world, settings = make_challenge_world("mask-cell-easy")
    assert world["medium"]["anomaly"]["kind"] == "mask-blocks"
    assert settings["method"] == "cell-search"
