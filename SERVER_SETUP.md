# Server setup and multi-machine coordination

How code, data and results move between the servers and how jobs are divided. **Everything is run from reg1**: Claude Code, editing, git, launching jobs, collecting results. No laptop is involved.

## Topology
```
GitHub (private, backup)  <--git push/pull-->  reg1 (hub)  --ssh: push_code / rsync / fanout-->  reg2 reg3 reg4 reg5
```
| Alias | Host | IP | `REG_SERVER_ID` | State (2026-10-26) |
|---|---|---|---|---|
| reg1 | bri-p7-1 | 10.206.10.43 | 0 | hub; this machine |
| reg2 | bri-p7-2 | 10.206.10.44 | 1 | SSH ok |
| reg3 | bri-p7-3 | 10.206.10.45 | 2 | SSH ok; venv already built (Phase 0 was started here) |
| reg4 | bri-p7-4 | 10.206.10.46 | 3 | SSH ok |
| reg5 | pending | pending | 4 | not delivered yet |

All four are Ubuntu with Python 3.12.3, one RTX 5000 Ada (32 GB), and an 11 TB `/data1` disk. No shared filesystem.

- **reg1 = hub**: the only machine you log in to. Holds the working repo at `/data1/moghadamp/reg-research` (symlinked as `~/reg`) and the data root `/data1/moghadamp/reg_data` (weights, datasets, text-embedding caches, all collected outputs).
- **reg2–5 = workers**: identical environment at the identical paths. They never contact GitHub or reg1; reg1 pushes code and data to them and pulls outputs back.
- **GitHub** = backup and history for code, configs, job files, notes, result CSVs. Only reg1 talks to it (works non-interactively from reg1). Never scp code.

## `env.sh` is the single description of the cluster
`source env.sh` in any shell; `fanout.sh`, `push_code.sh` and `sync_results.sh` source it themselves, on every host. It sets:

| Variable | Value | Notes |
|---|---|---|
| `REG_NUM_SERVERS` | 4 | **The one knob.** Change to 5 when reg5 is up. |
| `REG_HOSTS`, `REG_WORKERS` | `reg1 … regN`, `reg2 … regN` | derived |
| `REG_SERVER_ID` | hostname `bri-p7-N` → `N-1` | derived; preset it only on a host with a different naming scheme |
| `REG_DATA` | `/data1/moghadamp/reg_data` | same path on every server |
| `HF_HOME` | `$REG_DATA/weights/hf` | all HuggingFace downloads land here |

It also activates `.venv` if that server has one. Nothing cluster-specific goes in `~/.bashrc` or `~/.profile` (non-interactive SSH shells skip `~/.bashrc` on Ubuntu anyway, which is why the scripts source `env.sh` explicitly).

## Step 1 — SSH from reg1 to the workers (done for reg2–4)
`~/.ssh/config` on reg1 maps `reg2`–`reg4` to the IPs above with `User moghadamp`, `IdentityFile ~/.ssh/id_ed25519_reilly`, `IdentitiesOnly yes`. reg1 never SSHes to itself; `fanout.sh` runs the reg1 part locally.

Test: `bash scripts/fanout.sh --workers hostname`

**When reg5 arrives:** uncomment the `reg5` block in `~/.ssh/config` and fill in its IP; `ssh-copy-id -i ~/.ssh/id_ed25519_reilly reg5`; set `REG_NUM_SERVERS=5` in `env.sh`; copy tmux (Step 2); run Steps 3–5 for it. If its hostname is not `bri-p7-5`, extend the `case` in `env.sh`.

## Step 2 — tmux on the workers (done for reg2, reg4)
Workers have no system tmux and no sudo. reg1 has a self-contained copy under `~/.local` (`bin/tmux` wrapper + `bin/tmux.real` + `lib/libutempter*`), mirrored to each worker with rsync. `~/.profile` already puts `~/.local/bin` on `PATH`.

Check: `bash scripts/fanout.sh 'command -v tmux'`

## Step 3 — Deploy code (every time you commit)
```bash
bash scripts/push_code.sh
```
Creates (first run) or fast-forwards a checkout on each worker at the same path as on reg1, with a `~/reg` symlink. Uncommitted changes on reg1 are not sent. The worker checkouts are mirrors; never edit them there.

Then back up to GitHub: `git push`.

## Step 4 — Python environment, built on each server
`.venv` is built from `requirements.lock` (committed) by `scripts/setup_env.sh`. It is not copied between machines; each server builds its own, identically:
```bash
bash scripts/setup_env.sh                                            # reg1
bash scripts/fanout.sh --workers --parallel 'cd ~/reg && bash scripts/setup_env.sh'   # reg3 already has one; the script is a no-op there
```
Verify all servers report the same versions:
```bash
bash scripts/fanout.sh 'cd ~/reg && python -c "import torch,diffusers;print(torch.__version__,diffusers.__version__,torch.cuda.get_device_name())"'
```
(`python` resolves to `.venv/bin/python` because `env.sh` activates the venv.)

## Step 5 — Data and weights
Download **once on reg1** (`HF_HOME` already points into `$REG_DATA`), then mirror the read-only parts to the workers:
```bash
mkdir -p $REG_DATA/{weights,datasets,text_embeds,outputs,results}
bash scripts/fanout.sh --workers 'mkdir -p $REG_DATA/outputs'
for h in $REG_WORKERS; do rsync -a --info=progress2 $REG_DATA/weights $REG_DATA/datasets $REG_DATA/text_embeds $h:$REG_DATA/; done
```
Re-run the rsync whenever a new model or prompt set is added.

## Step 6 — Run the audit on all servers (Phase 0 step 1)
```bash
bash scripts/fanout.sh 'bash ~/reg/scripts/server_audit.sh' > audit_all.txt   # audit*.txt is gitignored
```
Paste the GPU/disk lines into `06-compute-and-datasets.md`.

## Step 7 — How a job runs across the servers
There is no scheduler. A job file is a list of rows; each server takes the rows where `idx % REG_NUM_SERVERS == REG_SERVER_ID`.

1. Create the job file (Claude Code or `scripts/make_jobs.py`) on reg1, commit, `bash scripts/push_code.sh`.
2. Start it everywhere, each inside a persistent `tmux` session:
   ```bash
   bash scripts/fanout.sh 'cd ~/reg && mkdir -p $REG_DATA/outputs/kill_or_go && tmux new -d -s job "python scripts/run_grid.py --jobs jobs/kill_or_go.jsonl --server-id $REG_SERVER_ID --num-servers $REG_NUM_SERVERS --out $REG_DATA/outputs/kill_or_go 2>&1 | tee -a $REG_DATA/outputs/kill_or_go/log_$(hostname -s).txt"'
   ```
   The single quotes keep `$REG_*` unexpanded on reg1; each host expands them after sourcing `env.sh`.
3. Check progress: `bash scripts/fanout.sh 'ls $REG_DATA/outputs/kill_or_go/*.json 2>/dev/null | wc -l; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'`
4. Attach if something looks wrong: `ssh reg3 -t tmux attach -t job` (on reg1 itself: `tmux attach -t job`).
5. The runner is resume-safe (skips rows whose output JSON exists), so after a crash just repeat step 2.

Changing `REG_NUM_SERVERS` reshuffles which server owns which row. Finish a job (or `sync_results.sh` and restart it everywhere) before adding reg5, because the resume check only sees the outputs already on each machine.

## Step 8 — Results flow back to reg1, then to git
- Images and per-image JSON: workers → reg1 with `bash scripts/sync_results.sh` (nightly in reg1's crontab: `0 3 * * * bash ~/reg/scripts/sync_results.sh >> ~/sync.log 2>&1`).
- Evaluation runs on reg1 over `$REG_DATA/outputs/<job>/` and writes `results/<job>.csv` into the repo.
- Commit the CSVs (small) and push to GitHub. Images never enter git.
- Backup: copy `$REG_DATA/results` and `$REG_DATA/outputs` to an external disk weekly.

## Step 9 — Where Claude Code runs
- **All development and orchestration:** on reg1, in `~/reg`, with GPU access. It drives the workers through `push_code.sh`, `fanout.sh` and `sync_results.sh`.
- **Long runs:** always in `tmux` (Step 7), never in an editor terminal that dies with the session.
- Workers never need interactive sessions; they only receive code and run job files.

## Daily routine (all on reg1)
1. Read `README.md` status; check the `fanout.sh` progress line.
2. Code changes → test on 5 prompts → commit → `bash scripts/push_code.sh` → `git push`.
3. New job file → Step 7.
4. Evening: `sync_results.sh`, run eval on finished batches, commit CSVs, update `06-*.md` GPU-hours.
