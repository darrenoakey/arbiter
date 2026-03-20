"""Tests for the Arbiter CLI."""
import json

import pytest

from arbiter.cli import main


class TestCLIParsing:
    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        # argparse exits with 0 on --help
        assert exc.value.code == 0

    def test_unknown_command(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["unknown-cmd"])
        assert exc.value.code == 2  # argparse returns 2 for invalid args

    def test_no_command(self, capsys):
        with pytest.raises(SystemExit):
            main([])
