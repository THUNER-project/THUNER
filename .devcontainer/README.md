# THUNER Claude Container

A containerized dev environment for running [Claude Code](https://docs.anthropic.com/claude/docs/claude-code) against the THUNER repo with `--dangerously-skip-permissions` enabled. The container provides the isolation boundary, so Claude can edit files, run commands, and install packages without affecting the host OS.

## Usage

### Option A — VS Code Dev Containers

1. Install the **Dev Containers** extension.
2. `Ctrl+Shift+P` → **Dev Containers: Reopen in Container**.
3. Once inside, open a terminal:
   ```bash
   claude --dangerously-skip-permissions
   ```

### Option B — Plain Docker CLI

Build the image (run from the repo root):

```bash
docker build -t thuner-claude .devcontainer
```

The container gets its own `.pixi` via the `thuner-pixi` named volume — separate from your host's `.pixi`, so their environments' baked-in absolute paths don't clobber each other. It starts empty, so build the environment into it once:

```bash
docker run -it --rm \
  -v ~/Documents/THUNER:/workspace/THUNER \
  -v thuner-pixi:/workspace/THUNER/.pixi \
  -v thuner-claude-local:/home/thuner/.local \
  -v ~/.claude:/home/thuner/.claude \
  -v ~/.claude.json:/home/thuner/.claude.json \
  -v ~/THUNER_output_claude:/home/thuner/THUNER_output \
  -w /workspace/THUNER \
  thuner-claude \
  bash -c "
    sudo chown thuner:thuner /workspace/THUNER/.pixi &&
    sudo chown thuner:thuner /home/thuner/.local &&
    pixi install
  "
```

The volume persists across runs, so afterwards just run Claude directly:

```bash
docker run -it --rm \
  -v ~/Documents/THUNER:/workspace/THUNER \
  -v thuner-pixi:/workspace/THUNER/.pixi \
  -v thuner-claude-local:/home/thuner/.local \
  -v ~/.claude:/home/thuner/.claude \
  -v ~/.claude.json:/home/thuner/.claude.json \
  -v ~/THUNER_output_claude:/home/thuner/THUNER_output \
  -w /workspace/THUNER \
  thuner-claude \
  claude --dangerously-skip-permissions
```

Or drop into a shell instead — swap `claude --dangerously-skip-permissions` for `bash`.

To explore a running container:
```bash
