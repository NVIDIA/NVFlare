# Authenticating Claude for the Skills Benchmark Harness

How to give the `skills_benchmark` Docker harness a working Claude credential and
run a benchmark, start to finish.

Harness location: `~/projects/NVFlare/dev_tools/agent/skills/benchmark`

## Background

The harness authenticates the in-container Claude Code via one of:

- An env var passed through to the container: `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, or `CLAUDE_CODE_OAUTH_TOKEN`.
- A mounted credentials file: host `~/.claude/.credentials.json` ->
  container `/workspace/.claude/.credentials.json` (mounted by default; disable
  with `--no-agent-auth-mount`).

`CLAUDE_CODE_OAUTH_TOKEN` is wired into `config/agents/claude.yaml`
(`passthrough_env`). It is forwarded at `docker run` time, so no image rebuild is
needed after adding it.

Note: running `/login` inside a managed/nested Claude Code session (where
`CLAUDECODE=1` / `CLAUDE_CODE_CHILD_SESSION=1`) authorizes the live session but
does NOT necessarily flush a reusable token to `~/.claude/.credentials.json`.
Use `claude setup-token` to mint a durable, reusable token instead.

## Step 1 - Get a durable token (in a plain terminal, NOT inside a managed session)

> Requires a Claude **Max or Pro** plan. If `setup-token` reports
> "Claude Max or Pro is required to connect to Claude Code", skip to the
> "Alternative: API key" section below and use an `ANTHROPIC_API_KEY` instead.

```bash
claude setup-token
```

A browser opens -> authorize with your power-user account -> it prints a token
like `sk-ant-oat01-...`. Copy that token.

## Step 2 - Export the token

```bash
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...paste-your-token-here...
```

Optional - make it permanent so you never repeat Step 1:

```bash
echo 'export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...' >> ~/.bashrc
```

## Step 3 - Confirm it's set

```bash
echo "${CLAUDE_CODE_OAUTH_TOKEN:0:12}...   (length=${#CLAUDE_CODE_OAUTH_TOKEN})"
```

You should see `sk-ant-oat01...` and a non-zero length. If empty, redo Step 2.

## Step 4 - Go to the benchmark dir

```bash
cd ~/projects/NVFlare/dev_tools/agent/skills/benchmark
```

## Step 5 - Run the benchmark

```bash
./bin/run.sh pair --agent claude --no-agent-auth-mount \
  --prompt /path/to/your/prompt.txt \
  /path/to/your/job-folder
```

- `--no-agent-auth-mount` skips mounting the empty `~/.claude/.credentials.json`
  so it cannot shadow the token.
- Replace `--prompt` and the job-folder path with your real paths.

## Step 6 - Check it worked

In the run's result directory:

```bash
cat host_report_status.json
cat console_output.log     # should NOT contain "Not logged in . Please run /login"
```

## Alternative: file-login flow instead of a token

Replace Steps 1-3 with: in a clean terminal run `claude` -> `/login` -> then
verify the credentials file is larger than 2 bytes:

```bash
wc -c ~/.claude/.credentials.json
```

If it is larger than 2 bytes, drop `--no-agent-auth-mount` in Step 5 and the
harness mounts the file automatically. If it is still 2 bytes, interactive login
is not persisting on this host - use the `setup-token` flow above instead.

## Alternative: API key (no Max/Pro subscription needed)

`claude setup-token` and the interactive `/login` subscription flow BOTH require
a Claude **Max or Pro** plan. If you see:

```text
Claude Max or Pro is required to connect to Claude Code
```

then the subscription path is not available to your account. Use an Anthropic
**API key** instead. This is billed per-token through the Anthropic Console
(pay-as-you-go) and is completely independent of Max/Pro. The harness already
forwards `ANTHROPIC_API_KEY` (it is in `passthrough_env`), so no harness edit is
needed for this path.

### What is an API key?

An API key (`sk-ant-api03-...`) authenticates direct calls to the Anthropic
API and is billed against Console credits, separate from any chat subscription.
It is the standard credential for headless / CI / container use like this
harness.

### Step 1 - Create an API key

Go to the Anthropic Console (account must have billing/credits set up):

- https://console.anthropic.com/settings/keys -> **Create Key** -> copy the
  `sk-ant-api03-...` value.

### Step 2 - Export it

```bash
unset CLAUDE_CODE_OAUTH_TOKEN          # avoid a stale subscription token competing
export ANTHROPIC_API_KEY=sk-ant-api03-...paste...
echo "len=${#ANTHROPIC_API_KEY}"       # should be > 0
```

Optional - make it permanent:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-api03-...' >> ~/.bashrc
```

### Step 3 - Run the benchmark

```bash
cd ~/projects/NVFlare/dev_tools/agent/skills/benchmark
./bin/run.sh pair --agent claude --no-agent-auth-mount \
  --prompt /path/to/your/prompt.txt \
  /path/to/your/job-folder
```

### Notes

- Billing: each `pair` run invokes the model inside the container and will incur
  API charges. Ensure the Console account has credits.
- `--no-agent-auth-mount` keeps the empty `~/.claude/.credentials.json` from
  shadowing the key.
- Check the result with `cat console_output.log` - it should NOT contain
  `Not logged in . Please run /login`.
