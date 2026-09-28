"""Claude Code PreToolUse hook (Section 4.10).

Install — add to .claude/settings.json:
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit",
      "hooks": [{"type": "command",
                 "command": "python -m multicomm_ensemble.adapters.claude_code_hook"}]
    }]
  }
}
The hook reads the tool-call JSON on stdin, routes code content through the
ensemble, and writes the (possibly transformed) JSON to stdout.
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from .. import config
from ..task import AgentTask
from ..pipeline import MultiLLMPipeline

# Verification side-effect: every invocation appends a line here so we can
# confirm Claude Code is actually running this PreToolUse hook.
_LOG_PATH = Path(__file__).with_name("claude_code_hook.log")


def _log(msg: str) -> None:
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except Exception:
        pass


def main() -> None:
    _log(f"invoked cwd={os.getcwd()!r} argv={sys.argv!r}")
    hook_input = json.load(sys.stdin)
    tool = hook_input.get("tool_name", "")
    ti = hook_input.get("tool_input", {})
    content = ti.get("content") or ti.get("new_string") or ""
    _log(f"parsed tool={tool!r} file={ti.get('file_path')!r} content_len={len(content)}")
    if tool in ("Write", "Edit") and len(content) > 10:
        task = AgentTask(
            description=f"Improve and verify this code before it is written:\n{content}",
            subtask_type="implement")
        result = MultiLLMPipeline().run_sync(task)
        print(f"[multicomm_ensemble] routed via {result.fusion_mode}, "
              f"provenance: {result.provenance}", file=sys.stderr)
        if config.mock_mode():
            # Mock proposals are canned and unrelated to the real task — never
            # let them overwrite real file content. Log only, pass through untouched.
            print("[multicomm_ensemble] MOCK MODE — content left untouched "
                  "(mock output would not match the real task)", file=sys.stderr)
        else:
            from ..exec_filter import extract_code_block
            fused_code = extract_code_block(result.fused_output) or content
            key = "content" if "content" in ti else "new_string"
            ti[key] = fused_code
        hook_input["_provenance"] = result.provenance
        _log(f"routed fusion_mode={result.fusion_mode!r} mock={config.mock_mode()}")
    else:
        _log("skipped (tool not Write/Edit or content_len<=10)")
    json.dump(hook_input, sys.stdout)


if __name__ == "__main__":
    main()