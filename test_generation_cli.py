import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import Generate


class GenerationCLITests(unittest.TestCase):
    def test_prompt_file_preserves_lines_and_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("Café\nThe morning ", encoding="utf-8-sig")
            self.assertEqual(Generate.read_prompt(prompt_file=path), "Café\nThe morning ")
        self.assertEqual(Generate.read_prompt(), "The")

    def test_blank_or_unreadable_prompt_has_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            for content in [b"  \n", b"\xff"]:
                path.write_bytes(content)
                with patch.object(sys, "argv", ["Generate.py", "--prompt-file", str(path)]), contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as error:
                    Generate.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("Could not read prompt", errors.getvalue())
            with self.assertRaises(OSError):
                Generate.read_prompt(prompt_file=Path(directory) / "missing.txt")

    def test_prompt_sources_are_mutually_exclusive(self):
        with patch.object(sys, "argv", ["Generate.py", "--prompt", "Hello", "--prompt-file", "prompt.txt"]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            Generate.main()
        self.assertEqual(error.exception.code, 2)

    def test_completion_only_uses_new_tokens_for_screen_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.txt"
            prompt.write_text("A starting line", encoding="utf-8")
            for completion_only in [False, True]:
                target = Path(directory) / f"result-{completion_only}.txt"
                tokenizer = Mock()
                tokenizer.encode.return_value = [1, 2]
                tokenizer.decode.return_value = "new text" if completion_only else "A starting line new text"
                args = ["Generate.py", "--prompt-file", str(prompt), "--output", str(target)]
                if completion_only:
                    args.append("--completion-only")
                with patch.object(sys, "argv", args), patch.object(Generate, "GPT"), patch("torch.load", return_value={}), patch("tiktoken.get_encoding", return_value=tokenizer), patch.object(Generate, "generate", return_value=torch.tensor([[1, 2, 3, 4]])), contextlib.redirect_stdout(io.StringIO()) as output:
                    Generate.main()
                tokenizer.encode.assert_called_once_with("A starting line")
                tokenizer.decode.assert_called_once_with([3, 4] if completion_only else [1, 2, 3, 4])
                self.assertEqual(target.read_text(encoding="utf-8"), output.getvalue())
                self.assertEqual(prompt.read_text(), "A starting line")


if __name__ == "__main__":
    unittest.main()
