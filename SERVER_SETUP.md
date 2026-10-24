# Server setup and 5-machine coordination

How files move and how the 5 servers divide work. Fill in the hostnames in the first step; every command after that uses the aliases.

## Topology
```
laptop (Windows, VS Code)  --ssh-->  reg1 (hub)  <--rsync-->  reg2 reg3 reg4 reg5
        |                              |
        +--- git push ---> remote <--- git pull on every server
```
- **Git remote** = source of truth for code, configs, job files, notes, result CSVs. Never scp code.
- **reg1 = hub**: holds `/data/reg/` (datasets, model weights, text-embedding caches) and collects all generated images under `/data/reg/outputs/`.
- **reg2–5 = workers**: identical environment, pull code from git, pull data from reg1, push outputs back to reg1.
- If the servers already share an NFS mount, put `/data/reg` on it and skip every rsync step below.

## Step 1 — SSH from the laptop (once, 15 min)
In Git Bash on the laptop:
```bash
ssh-keygen -t ed25519 -C "reg-laptop"          # accept defaults, set a passphrase
for h in reg1 reg2 reg3 reg4 reg5; do ssh-copy-id USER@$h; done   # or paste ~/.ssh/id_ed25519.pub into ~/.ssh/authorized_keys on each
```
Add to `~/.ssh/config` on the laptop (edit `HostName`/`User`):
```
Host reg1
    HostName 10.0.0.11
    User pmoghadam
Host reg2
    HostName 10.0.0.12
    User pmoghadam
Host reg3
    HostName 10.0.0.13
    User pmoghadam
Host reg4
    HostName 10.0.0.14
    User pmoghadam
Host reg5
    HostName 10.0.0.15
    User pmoghadam
Host reg*
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ForwardAgent yes
```
Test: `ssh reg1 hostname`.

Then let reg1 reach the workers without a password (needed for rsync):
```bash
ssh reg1 'ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 <<<y >/dev/null; cat ~/.ssh/id_ed25519.pub'
# paste that key into ~/.ssh/authorized_keys on reg2–reg5, and copy the same Host block into reg1's ~/.ssh/config
```

## Step 2 — Git remote (once, 10 min)
**Option A (preferred):** private GitHub/GitLab repo, if institutional policy allows. Nothing sensitive goes in the repo (no patient data, no images).

**Option B (no external service):** a bare repo on reg1:
```bash
ssh reg1 'mkdir -p ~/git && git init --bare ~/git/reg.git'
```
On the laptop, in the repo folder:
```bash
git init -b main
git add .
git commit -m "Initial research notes, plan, server scripts"
git remote add origin reg1:git/reg.git          # Option B; for Option A use the https/ssh URL
git push -u origin main
```
On every server:
```bash
git clone reg1:git/reg.git ~/reg     # or the GitHub URL
```

## Step 3 — Run the audit on all 5 (Phase 0 step 1)
From the laptop:
```bash
bash scripts/fanout.sh 'bash ~/reg/scripts/server_audit.sh' > audit_all.txt
```
Paste the GPU/disk/mount lines into `06-compute-and-datasets.md`. **Check now whether a shared mount exists** (section "SHARED STORAGE"); it changes Step 4.

## Step 4 — Data and weights live on reg1
```bash
ssh reg1 'sudo mkdir -p /data/reg/{weights,datasets,text_embeds,outputs,results} && sudo chown -R $USER /data/reg'
```
Environment on every server (append to `~/.bashrc`):
```bash
export REG_DATA=/data/reg
export HF_HOME=/data/reg/weights/hf      # all HuggingFace downloads land here
export HF_HUB_ENABLE_HF_TRANSFER=1
```
Download weights and datasets **once on reg1**, then mirror to workers:
```bash
bash scripts/fanout.sh --workers 'mkdir -p /data/reg'      # workers need the same path
ssh reg1 'for h in reg2 reg3 reg4 reg5; do rsync -a --info=progress2 /data/reg/weights /data/reg/datasets /data/reg/text_embeds $h:/data/reg/; done'
```
Re-run that rsync whenever a new model or prompt set is added. (With NFS: skip, everything is already visible everywhere.)

## Step 5 — Python environment, built once and cloned
Phase 0 step 2 builds the env on reg1 and pins it in `environment.yml` (committed). Workers then run:
```bash
cd ~/reg && git pull && conda env create -f environment.yml   # or: uv sync
```
Verify all five report the same versions:
```bash
bash scripts/fanout.sh 'cd ~/reg && conda run -n reg python -c "import torch,diffusers;print(torch.__version__,diffusers.__version__,torch.cuda.get_device_name())"'
```

## Step 6 — How a job runs across 5 machines
There is no scheduler. A job file is a list of rows; each server takes the rows where `idx % 5 == server_id`.

1. Create the job file (Claude Code or `scripts/make_jobs.py`), commit, push.
2. Start it everywhere, each inside a persistent `tmux` session:
   ```bash
   bash scripts/fanout.sh 'cd ~/reg && git pull -q && tmux new -d -s job "conda run -n reg python scripts/run_grid.py --jobs jobs/kill_or_go.jsonl --server-id \$REG_SERVER_ID --out /data/reg/outputs/kill_or_go 2>&1 | tee -a /data/reg/outputs/kill_or_go/log_\$(hostname).txt"'
   ```
   `REG_SERVER_ID` is 0–4, set once per server in `~/.bashrc` (`export REG_SERVER_ID=0` on reg1, … `=4` on reg5).
3. Check progress: `bash scripts/fanout.sh 'ls /data/reg/outputs/kill_or_go/*.json 2>/dev/null | wc -l; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'`
4. Attach to a server if something looks wrong: `ssh reg3 -t tmux attach -t job`.
5. The runner is resume-safe (skips rows whose output JSON exists), so after a crash just repeat step 2.

## Step 7 — Results flow back to reg1, then to git
- Images and per-image JSON: workers → reg1 via `scripts/sync_results.sh` (run from the laptop or reg1; also put it in reg1's crontab nightly: `0 3 * * * bash ~/reg/scripts/sync_results.sh`).
- Evaluation runs on reg1 over `/data/reg/outputs/<job>/` and writes `results/<job>.csv` into the repo.
- Commit the CSVs (small) and push. Images never enter git.
- Backup: copy `/data/reg/results` and `/data/reg/outputs` to an external disk weekly (`rsync -a reg1:/data/reg/results /path/to/backup/`).

## Step 8 — Where Claude Code runs
- **Interactive development:** VS Code on the laptop → *Remote-SSH: Connect to Host* → `reg1` → open `~/reg`. The Claude Code extension then runs on reg1 with GPU access. This is the normal way to do Phase 0 steps 2–9 and all coding.
- **Long runs:** always in `tmux` (step 6), never in a VS Code terminal that dies when the laptop sleeps.
- Workers never need interactive sessions; they only `git pull` and run job files.

## Daily routine
1. Laptop/reg1: `git pull`, read `README.md` status, check `fanout.sh` progress line.
2. Code changes on reg1 → test on 5 prompts → commit → push → `fanout.sh 'cd ~/reg && git pull'`.
3. New job file → step 6.
4. Evening: `sync_results.sh`, run eval on finished batches, commit CSVs, update `06-*.md` GPU-hours.
