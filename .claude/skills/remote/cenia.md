# CENIA

The SLURM cluster this project has used. Official documentation: https://cluster.cenia.cl/ (hardware, usage policy, SLURM policies, storage, guide, FAQ); reread it when something here looks outdated.

## Access and setup (once per user)
- An account on the cluster (requested through CENIA) and an SSH key. In `~/.ssh/config`:
  ```
  Host cenia
    HostName <login-host>   # from the cluster documentation
    Port <ssh-port>
    User <your-cluster-user>
    IdentityFile ~/.ssh/<your-key>
  ```
  `ssh cenia` reaches the login node kraken; compute nodes are reachable with `ProxyJump cenia`, but work on them only through jobs.
- On kraken: `git clone` the repository into `~/<repo>`, `mkdir logs`, `cp .claude/skills/remote/cenia/slurm.ini src/remote/slurm.ini` (the profile: partition `ialab-high`, node llaima, one A40, account and QoS), and copy `.env` with `scp`, adding to it:
  ```
  WORK_DIR=$HOME/scratch/<project>          # virtual environments and pip cache
  MODEL_CACHE=/workspace1/<group>/$USER/<project>/hf   # llaima's local workspace (cluster rule: models are read from workspaces only)
  ```
  where `<group>` is your group's folder under `/workspace1` (ask the lab).
- The first job (`setup`) builds the environment and downloads the weights into `MODEL_CACHE`; every other job depends on it.

## Partitions and nodes
| Partition | Nodes (GPUs) | Use |
|---|---|---|
| `ialab-high` | llaima (8 A40 48 GB), antuco (2 AMD MI210 64 GB, ROCm) | first choice |
| `ialab-low` | hydra (8 Titan RTX 24 GB), ventress (8 RTX 2080 Super 8 GB), scylla (1080 Ti and 2080 Ti, 11 GB), yodaxico (8 1080 Ti 11 GB), ahsoka | when `ialab-high` is full and the job fits |
| `*-unlimit` variants | the same nodes | only with QoS `debug` (1 h, one running job); they share the GPUs of the base partitions |
- The profile pins llaima: the models need 48 GB and bfloat16, and its workspace holds the weights; move jobs elsewhere only on the user's decision.
- `ialab-low` caps: at most 4 CPUs and 20 GB of RAM per GPU; Titan RTX (Turing) has no bfloat16 or FlashAttention 2. Antuco needs ROCm wheels and `unset CUDA_VISIBLE_DEVICES`.

## Rules and limits (QoS `regular`)
- Every job: `--account=default-account`, `--qos=regular`, a partition and a descriptive `--comment` (the profile sets them).
- Per user, across all partitions: 24 h per job, at most 4 GPUs per job, 4 running and 32 submitted jobs. CPU-only jobs: `--gres=none` and `--mem` within one GPU's share of the node.
- Kraken is for editing, `git`, `sbatch` and light commands only: nothing heavy, not even a recursive `du`, `find` or `rm -rf` of a big tree (do inventories and mass deletions inside a small CPU job, `--cpus-per-task=1 --mem=4G`).
- Long campaigns (many jobs, or jobs over a day) are announced in the cluster's Telegram group by the account holder: remind the user before launching one; individual jobs under 24 h need no announcement. Needs beyond the limits go through the additional-resources form of the documentation site.
- Penalties escalate: a warning and cancellation, then a one-week block, then longer blocks or the loss of the account.

## Priority
Priority = QoS (1,000,000) + fair-share (100,000, half-life 7 days: heavy recent use lowers it) + job size (10,000) + age (1,000); `scontrol top` is not allowed. Free GPUs per node: `scontrol show node llaima | grep -E 'Gres|AllocTRES'` (free = Gres minus `gres/gpu` in AllocTRES) and `squeue -w llaima`; never infer them from one partition's queue. A pending job with reason `Priority`, `Resources` or `ReqNodeNotAvail` is waiting for llaima; a limit shows as `AssocMaxJobsLimit`, `QOSMax...` or a held job.

## Storage
| Space | Path | Size | Notes |
|---|---|---|---|
| home | `~` (same on every node) | 50 GB | code, configs, small results; backed up |
| scratch | `~/scratch` | 500 GB | environments, caches, large results; backed up |
| archive | `~/archive` | 200 GB | finished results kept long term; backed up |
| workspace | `/workspace1/<group>/$USER` on each node (from kraken, `/workspaces/<node>-workspace1/<group>/$USER`) | 200 GB per node | the node's local disk: fast, not backed up, shared with every user of the node |
- Model weights and training data are read only from the workspace of the node the job runs on; each node's workspace is separate, so a job sent to a new node must fill that node's cache first.
- Check free space before downloading into a workspace (`df -h /workspaces/<node>-workspace1` from kraken): a full workspace breaks every job on that node. Measure our usage with a CPU job (`du -h --max-depth=2`) and delete caches and abandoned downloads periodically.

## Tools
`sfree` (free resources per node), `sq` (our queue), `sshow <id>`, `stail <id>`, `serror <id>`, `seff <id>` (efficiency of a finished job), `sprio -j <id>`, `sshare -u $USER`. Log files start with a banner (lines starting with "Puedes", "(y sug", "en tu", "Dudas", "----"); strip it when reading.
