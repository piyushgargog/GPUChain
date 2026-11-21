# Experiments

Scripts that measure what Ekido claims, and the raw results behind every number. A number in a README or on the website must trace back to a file in `experiments/results/`.

Results from a synthetic workload or a CPU are labelled as such in their `evidence_type` field and are not GPU evidence.

## Quality comparison: resumed vs uninterrupted

Does resuming with a different micro-batch hurt the final model?

For each seed the script trains one uninterrupted run at micro-batch 32, then starts a second process from that run's step-150 checkpoint at micro-batch 8 x 4 accumulation and trains to the end. Both final checkpoints are evaluated on the same examples with the verifier's own evaluation code, and the difference is resumed accuracy minus baseline accuracy.

```bash
# Checks the script on CPU in a few minutes. Says nothing about the real model.
python experiments/quality_comparison.py --synthetic --seeds 2 --steps 300 --resume-step 150 --eval-examples 512

# The real run, on a GPU. Pin the model and dataset first.
python experiments/prepare_identity.py --out experiments/identity_agnews.json
python experiments/quality_comparison.py --seeds 5 --device cuda --identity experiments/identity_agnews.json
```

The real run needs `transformers`, `peft`, `datasets` and `huggingface_hub` on top of the runtime's dependencies. Ask the team lead before a long GPU run on a laptop.

### Result

Raw file: [`results/external_vaibhav_t4_20261106/quality_comparison_20261105_181231.json`](results/external_vaibhav_t4_20261106/quality_comparison_20261105_181231.json). Tesla T4, FP16, DistilBERT LoRA on AG News, 400 logical steps, resumed from step 150, 2,000 evaluation examples, seeds 1 to 5.

| Seed | Baseline accuracy | Resumed accuracy | Difference |
|---:|---:|---:|---:|
| 1 | 0.9020 | 0.9055 | +0.0035 |
| 2 | 0.9015 | 0.9030 | +0.0015 |
| 3 | 0.9035 | 0.9040 | +0.0005 |
| 4 | 0.9020 | 0.9000 | -0.0020 |
| 5 | 0.8995 | 0.9005 | +0.0010 |

Mean difference +0.0009 (+0.09 accuracy points), 95% CI [-0.0016, +0.0034], worst seed -0.0020.

**Method:** the interval is a two-sided 95% t-interval over the five per-seed differences: mean ± 2.776 x standard deviation / sqrt(5), with 4 degrees of freedom (`seed_statistics` in `quality_comparison.py`). The script writes a claim only at 5 or more seeds and 2,000 or more examples on the real model, and only in the form "no degradation detected beyond 2 accuracy points at this sample size".

### What this does not show

- Both arms ran on one T4. It compares a changed micro-batch, not two kinds of GPU.
- No process was killed. The baseline finished, and the resumed arm started from its saved step-150 checkpoint.
- That run used model revision `main` and an all-zero dataset fingerprint, so its inputs are not pinned. `--identity` was added afterwards for new runs.
- The files were supplied by the person who ran them. The folder's `PROVENANCE.md` says what was and was not checked on receipt.
- Dropout is left on, as in the demo job, so different micro-batches also draw different dropout masks.

## Cross-GPU resume: T4 -> RTX 3060 -> T4

Can a checkpoint written on one kind of GPU be restored on another and keep training?

One job was carried by hand between two machines. Each leg restores the previous leg's checkpoint, re-checks the canary loss, trains on, saves, and writes an `evidence.json` describing the machine it really ran on.

```bash
# Checks the script on CPU. Labelled as a script check, not evidence.
python experiments/cross_gpu_resume.py start --synthetic --out handover --steps 150 --device cpu
python experiments/cross_gpu_resume.py resume --checkpoint handover --out from_other_gpu --steps 50 --micro-batch 8 --device cpu

# The real thing, one command per GPU.
python experiments/cross_gpu_resume.py start --out handover --steps 150 --identity experiments/identity_agnews.json
python experiments/cross_gpu_resume.py resume --checkpoint handover --out from_other_gpu --steps 50 --micro-batch 8
python experiments/cross_gpu_resume.py resume --checkpoint from_other_gpu --out back_home --steps 50 --micro-batch 16
```

`infra/cross_gpu_notebook.ipynb` runs one resume leg on a hosted T4. Upload a zip of this repository and the zipped checkpoint folder from the previous leg, set the two file names in its second cell, and run every cell. It pins `transformers==5.22.1` and `peft==0.21.2`. Never put a wallet key or `.env` in a notebook.

### Result

One run, job seed 1337, DistilBERT LoRA on AG News, logical batch 32, accuracy on 2,000 evaluation examples.

| Leg | GPU | Steps | Precision | Micro-batch x accumulation | Accuracy | Canary saved / measured | Raw files |
|---|---|---|---|---|---|---|---|
| 1 | Tesla T4 | 0 to 150 | FP16 | 16 x 2 | 0.896 | none, first leg | [`results/external_t4_leg1_20261105/`](results/external_t4_leg1_20261105/) |
| 2 | RTX 3060 Laptop | 150 to 200 | BF16 | 8 x 4 | 0.906 | 0.043106809 / 0.043106779 | [`results/leg2_3060_20261107/`](results/leg2_3060_20261107/) |
| 3 | Tesla T4 | 200 to 250 | FP16 | 16 x 2 | 0.905 | 0.066218399 / 0.066218361 | [`results/external_vaibhav_leg3_20261107/`](results/external_vaibhav_leg3_20261107/) |

The canary saved for leg 3 is the `canary.loss_fp32` in leg 2's step-200 manifest, which ties leg 3 to the checkpoint the 3060 wrote. The job spec and its `immutable_hash` are the same in all three manifests.

**Versions:** legs 1 and 3 ran torch 2.11.0+cu130, Python 3.13.16, CUDA 13.0 on Linux; leg 2 ran torch 2.14.1+cu130, Python 3.14.5, CUDA 13.0 on Windows. Each `evidence.json` records its own, and leg 3's `environment.json` lists every package version.

### What this does not show

- That the result equals an uninterrupted run. The quality comparison above measures that, on one GPU.
- A trend in accuracy. Differences of one point on 2,000 examples are within sampling noise.
- A process kill. Each leg was a planned resume from a saved checkpoint.
- More than one run or one seed.
- That legs 1 and 3 used the same physical T4. They were separate hosted sessions.
- An unbroken hash chain between legs. The script keeps only each leg's newest checkpoint, so the manifests at steps 175 and 225 were not kept.
- Pinned inputs. The spec uses model revision `main` and an all-zero dataset fingerprint.
- Checkpoint tensors. Only manifests, specs and evidence are committed, so the checkpoints can't be re-validated from this repository.

## Mid-training margin: how far does honest training dip?

A probation check compares each checkpoint with the one before it and refuses the provider if accuracy fell by more than a margin. Training is noisy, so an honest run can dip between checkpoints. This measures how often an honest run would be refused at each margin.

The script trains honest runs only, evaluates every checkpoint against the previous one with the verifier's evaluation code, and counts a pair as refused when the one-sided 95% lower bound on the accuracy difference is below minus the margin.

```bash
# Synthetic workload on CPU. Says nothing about the real model.
python experiments/probation_noise.py --seeds 8

# The real model, on a GPU.
python experiments/probation_noise.py --real --seeds 5 --device cuda --eval-examples 2000 --identity experiments/identity_agnews.json
```

### Result

Raw file: [`results/external_vaibhav_probation_20261107/probation_noise_real_20261107_074433.json`](results/external_vaibhav_probation_20261107/probation_noise_real_20261107_074433.json). Tesla T4, DistilBERT LoRA on AG News with pinned model and dataset revisions, 400 logical steps per seed, a checkpoint every 25 steps, 2,000 evaluation examples, seeds 1 to 5. That gives 15 consecutive pairs per seed, 75 in all.

| Margin | Honest pairs refused | Seeds with any refusal |
|---|---|---|
| 0.02 | 0 of 75 | 0 of 5 |
| 0.03 | 0 of 75 | 0 of 5 |
| 0.05 | 0 of 75 | 0 of 5 |
| 0.08, 0.10, 0.15 | 0 of 75 | 0 of 5 |

Worst pair: seed 4, step 275 to 300, accuracy 0.9035 to 0.8930, difference -0.0105, lower bound -0.0165.

**Chosen margin: 0.05.** The mid-training margin stays at 0.05, which leaves about 3.4 points between it and the worst lower bound seen here. The write-up is [`results/external_vaibhav_probation_20261107/PROBATION_RESULT_NOTE.md`](results/external_vaibhav_probation_20261107/PROBATION_RESULT_NOTE.md).

For comparison, the earlier synthetic measurement on CPU ([`results/probation_noise_synthetic_20261105_215234.json`](results/probation_noise_synthetic_20261105_215234.json), 8 seeds, 232 pairs) refused 2 pairs at 0.02 and none at 0.03 or above. It is a different workload and is not evidence about the real model.

### What this does not show

- That 0.05 is right for other seeds, GPUs, job lengths or evaluation sets. Five seeds is a small sample, and zero refusals in 75 pairs does not mean the rate is zero.
- Anything about catching a bad provider. Only honest runs were trained, so this is a false-refusal test, not a detection test.
- Cross-GPU behaviour or a process kill. One T4, uninterrupted runs.
- The run log. The note refers to `probation_run.log`, which is not committed because `*.log` is ignored.

## Fault matrix on a GPU: status

`fault_matrix.py` injects one fault at a time into a running demo stack and checks what actually happened: the failure cause, the failover, the payouts and refund, and that no container is left holding the GPU. Each fault gets its own funded job, and the results file is rewritten after every fault.

```bash
python experiments/demo_stack.py --total-steps 300      # in one terminal, wait for READY
python experiments/fault_matrix.py --faults OOM         # in another, one fault per run
```

Per the contributing rules, ask the team lead before a long GPU run on a laptop. The script does not check free RAM or GPU temperature. Check both by hand (for example `nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv` and the machine's free memory) before each fault, and run one fault per command so a failure doesn't hide the others.

### Where each fault stands

| Fault | End-to-end test on CPU | Run on a GPU |
|---|---|---|
| `OOM` | `test_oom_fails_over_and_is_labelled` | Did not run for this issue. One earlier attempt, below, did not pass its own check |
| `CRASH_MID_CHECKPOINT` | `test_crash_while_writing_a_checkpoint_resumes_from_the_one_before` | Did not run |
| `CORRUPT_CHECKPOINT` | `test_corrupt_checkpoint_is_refused_and_fails_over` | Did not run |
| `DISCONNECT` | `test_disconnect_fails_over_and_the_old_worker_is_stopped` | Did not run |
| `HEARTBEAT_LOSS` | `test_silent_worker_is_declared_dead_and_fenced_when_it_returns` | Did not run |
| `THERMAL_WARNING` | `test_thermal_warning_saves_a_checkpoint_without_failing_over` | One earlier run, below. Its file doesn't record the device |

The tests are in `tests/test_e2e_faults.py`. "Did not run" means no raw result file exists, not that the fault is untested: each has been exercised end to end on CPU.

### Why the GPU runs did not happen

The six faults were not re-run on a GPU for this issue. The recovery behaviour they touch already has measured results: a provider killed mid-run on a real GPU, and a replacement that returned junk and was refused. Those are listed in the main README with their raw files. They cover a killed provider and bad work, which are not the six faults above, so they do not replace a per-fault GPU result. No free RAM or GPU temperature readings exist because nothing was run.

If a GPU run of these faults is wanted later, use the commands above, one fault per command, and check free RAM and `nvidia-smi` temperature before each.

### The one earlier attempt

Raw file: [`results/fault_matrix_20261105_005648.json`](results/fault_matrix_20261105_005648.json), from 5 Nov. It holds two faults and nothing else was run in that session.

- `THERMAL_WARNING`: passed its checks. The warning reached the event stream, no failover happened, the job settled with 300 of 300 steps accepted.
- `OOM`: **did not pass**. The first failure the coordinator recorded was `HEARTBEAT_LOST`, where the check expects `OOM`. The job took 3 attempts and 396 s after the fault, one checkpoint step (150) was rejected, and it still settled.

The file records no device, GPU name or driver, so it is not evidence of a GPU run. Its provider names ("Provider B (capped 768 MiB)") match the capped providers `demo_stack.py` describes for the GPU stack, which is a reason to suspect it ran there, not proof.

## Files

| Path | What |
|---|---|
| `quality_comparison.py` | The comparison above. Also provides the job spec and run helper the other experiments import |
| `prepare_identity.py` | Resolves the model and dataset commits and hashes the full AG News splits, for new real runs |
| `cross_gpu_resume.py` | One leg of the cross-GPU test: start or resume, then write the evidence |
| `../infra/cross_gpu_notebook.ipynb` | Runs one resume leg on a hosted T4 |
| `results/external_t4_leg1_20261105/`, `results/leg2_3060_20261107/`, `results/external_vaibhav_leg3_20261107/` | The three legs of the round trip, with provenance notes |
| `fault_matrix.py` | Injects each fault into a running demo stack, one at a time, and checks the outcome |
| `probation_noise.py` | The margin measurement above |
| `results/external_vaibhav_probation_20261107/` | The real-model margin run, the identity it was pinned to, and its write-up |
| `results/probation_noise_synthetic_20261105_215234.json` | The earlier synthetic margin measurement, CPU only |
| `results/external_vaibhav_t4_20261106/` | The T4 quality comparison, the T4 start-leg metadata sent with it, and notes on how they were produced |

## Platform API load test at stage-1 rates (k6)

Ticket P9 (ekido-ai/ekido-internal#66). Script: [`k6/platform-stage1.js`](k6/platform-stage1.js). Raw result: [`results/scale/k6-2026-11-22.json`](results/scale/k6-2026-11-22.json).

**Rate.** Stage 1 is 1,000 teams. [14-scaling-design.md](https://github.com/ekido-ai/ekido-internal/blob/main/docs/ekido/14-scaling-design.md) §3.7 gives stage 2 (100,000 teams) "tens of requests per second on average, hundreds at peak". Scaled by 1/100 that's ~0.3 req/s average and ~3 req/s at peak for stage 1 *(estimate)*. The run used **6 req/s in total, twice that peak, for 5 minutes**: 1 req/s each to `/v1/me`, the team page, the key page, the job list and `/v1/recipes`, and 0.5 req/s each to sign-in and quotes.

```bash
# A test account on staging: SESSION is its ekido_session cookie, ORG its team id. Never commit either.
SESSION=… ORG=… OUT=experiments/results/scale/k6-$(date +%F).json k6 run experiments/k6/platform-stage1.js
```

Sign-in is measured by `POST /v1/auth/email/verify` with a wrong token. That runs the sign-in lookup without sending an email, and a 4xx is the expected answer. The platform has no customer job list yet, so "the job list" is the coordinator's `GET /sepolia/jobs?limit=40`, the one the dashboard and admin pages read.

### Result (22 Nov 2026 01:00 IST, i.e. 21 Nov 19:30 UTC; staging; client on a home network in India)

| Endpoint | Requests | p50 | p95 | p99 | Errors |
|---|---:|---:|---:|---:|---:|
| `POST /v1/orgs/{org}/quotes` | 151 | **61 ms** | **75 ms** | 300 ms | 0 |
| `GET /v1/orgs/{org}` (team) | 301 | 54 ms | 72 ms | 332 ms | 0 |
| `POST /v1/auth/email/verify` (sign-in) | 150 | 53 ms | 72 ms | 206 ms | 0 |
| `GET /v1/me` | 301 | 53 ms | 66 ms | 162 ms | 0 |
| `GET /v1/orgs/{org}/api-keys` (keys) | 301 | 54 ms | 64 ms | 247 ms | 0 |
| `GET /sepolia/jobs?limit=40` (job list) | 301 | 44 ms | 53 ms | 99 ms | 0 |
| `GET /v1/recipes` | 301 | 39 ms | 50 ms | 323 ms | 0 |

1,806 requests, no errors, every check passed. Stage-1 load is well within what one server handles.

**Slowest: quotes, and why.** These latencies include the client's network: one round trip to staging was 37 ms (median of 30 TCP connects, in the JSON), and k6 reuses connections. Taking that off gives a rough server time:
- `/v1/recipes`: ~2 ms. A static list with no database.
- Job list: ~7 ms.
- Signed-in reads (`me`, team, keys) and sign-in: ~16 ms. Each first looks up the session or token in Postgres (`user_for_session`), then runs its own query. The design's Valkey cache for sessions and keys (§4.3) isn't in yet.
- **Quotes: ~24 ms.** It is the only write in the mix. It looks up the session, loads the dataset row with its stored rows, builds the quote (hashing the dataset and the terms), then `INSERT`s and `COMMIT`s a quote row.

The p99 and max values (up to 1.8 s on `/v1/me`) are single outliers across all endpoints, with no pattern, so they are more likely the home network than the server. A run from a machine in the same region as staging would tell them apart.

**Limits of this run:** one client, one test account and one dataset; staging's deployed commit isn't exposed (platform `main` was 7891c63 at the time); the 151 quotes it wrote are test rows on staging.
