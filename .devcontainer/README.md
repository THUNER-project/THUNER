# THUNER Claude Container

A containerized dev environment for running [Claude Code](https://docs.anthropic.com/claude/docs/claude-code)
against the THUNER repo with `--dangerously-skip-permissions` enabled. The container
provides the isolation boundary, so Claude can edit files, run commands, and install
packages without affecting the host OS.

## Usage

### Option A — VS Code Dev Containers

1. Install the **Dev Containers** extension.
2. `Ctrl+Shift+P` → **Dev Containers: Reopen in Container**.
3. Once inside, open a terminal:
   ```bash
   claude --dangerously-skip-permissions
   ```

### Option B — Plain Docker CLI

Build (run from the repo root):

```bash
docker build -t thuner-claude .devcontainer
```

Run Claude directly:

```bash
docker run -it --rm \
  -v ~/Documents/THUNER:/workspace/THUNER \
  -v ~/.claude:/home/thuner/.claude \
  -v ~/THUNER_output_claude:/home/thuner/THUNER_output \
  -w /workspace/THUNER \
  thuner-claude \
  claude --dangerously-skip-permissions
```

Or drop into a shell instead:

```bash
docker run -it --rm \
  -v ~/Documents/THUNER:/workspace/THUNER \
  -v ~/.claude:/home/thuner/.claude \
  -v ~/THUNER_output_claude:/home/thuner/THUNER_output \
  -w /workspace/THUNER \
  thuner-claude \
  bash
```