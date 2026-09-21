# Online Evals on EC2 — Setup Steps

Deploys `src/evals/online_evals/run_online_eval.py` to a separate EC2 instance and runs it as a cron job every 15 minutes. Each run scores new Langfuse traces (`answer_relevancy`, `faithfulness`, `contextual_relevancy`) and pushes the scores back to Langfuse.

**Layout on the instance**

```
/opt/campusx-rag/
├── .env                      # created from Secrets Manager (chmod 600)
├── venv/                     # Python virtualenv
├── run_eval.sh               # wrapper called by cron (cd + flock + log)
└── online-evals/
    ├── run_online_eval.py    # pulled from S3
    ├── requirements.txt      # pulled from S3
    ├── eval_checkpoint.json  # created on first run; do not delete
    └── cron.log              # script output
```

## Prerequisites

- EC2 instance running Ubuntu (t3.small, ~15 GiB disk is enough). Only port 22 needs to be open — nothing listens.
- IAM instance profile attached to the instance with:
  - `secretsmanager:GetSecretValue` on `api-keys-*`
  - `s3:GetObject` and `s3:ListBucket` on `campusx-rag-online-evals`
- Secret `api-keys` (region `ap-south-1`) containing `OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL`.
- S3 bucket `campusx-rag-online-evals`.

## 1. Local machine (PowerShell): upload to S3

The script calls `client.api.observations.get_many(...)`, which depends on the Langfuse SDK version, so pin the same versions used locally.

```powershell
uv pip freeze | Select-String -Pattern '^(deepeval|langfuse|python-dotenv)==' | ForEach-Object Line | Set-Content requirements.txt
```

```powershell
aws s3 cp requirements.txt s3://campusx-rag-online-evals/requirements.txt --region ap-south-1
```

```powershell
aws s3 cp src/evals/online_evals/run_online_eval.py s3://campusx-rag-online-evals/run_online_eval.py --region ap-south-1
```

## 2. EC2 (Ubuntu): one-time setup

Paste this whole block. It is safe to re-run.

```bash
set -e
R=/opt/campusx-rag
S3=s3://campusx-rag-online-evals
REGION=ap-south-1

# packages (+ AWS CLI v2 if missing)
sudo apt-get update
sudo apt-get install -y jq unzip cron python3-venv python3-pip
command -v aws >/dev/null || { curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip && unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install; }

# folders (owned by you so the .env redirect works without sudo)
sudo mkdir -p $R/online-evals
sudo chown -R $USER:$USER $R

# script + libraries
aws s3 cp $S3/run_online_eval.py $R/online-evals/ --region $REGION
aws s3 cp $S3/requirements.txt   $R/online-evals/ --region $REGION
python3 -m venv $R/venv
$R/venv/bin/pip install -q -r $R/online-evals/requirements.txt

# .env from Secrets Manager
aws secretsmanager get-secret-value --secret-id api-keys --region $REGION \
  --query SecretString --output text \
  | jq -r 'to_entries[] | "\(.key)=\(.value)"' > $R/.env
chmod 600 $R/.env

# wrapper: cd + no-overlap lock + log
cat > $R/run_eval.sh <<'EOF'
#!/bin/bash
cd /opt/campusx-rag/online-evals
exec /usr/bin/flock -n /tmp/online_eval.lock /opt/campusx-rag/venv/bin/python run_online_eval.py >> cron.log 2>&1
EOF
chmod +x $R/run_eval.sh

# cron every 15 min (won't duplicate on re-run)
sudo systemctl enable --now cron
( crontab -l 2>/dev/null | grep -v run_eval.sh; echo '*/15 * * * * /opt/campusx-rag/run_eval.sh' ) | crontab -
```

What the wrapper and cron line do:

| Piece | Purpose |
|---|---|
| `cd .../online-evals` | Run from the script folder; `load_dotenv()` walks up from here to find `/opt/campusx-rag/.env`, and the checkpoint file is written here. |
| `flock -n /tmp/online_eval.lock` | If the previous run is still going, skip this tick instead of overlapping. |
| `venv/bin/python` (absolute path) | Cron has a minimal `PATH` and does not activate the venv. |
| `>> cron.log 2>&1` | Append stdout and stderr (Python `logging` writes to stderr) to the log. |
| `*/15 * * * *` | Every 15 minutes: :00, :15, :30, :45. |

## 3. Verify

Run once by hand:

```bash
/opt/campusx-rag/run_eval.sh
```

Read the result. The last line should be `Run complete: fetched=… scored=…`:

```bash
tail -n 20 /opt/campusx-rag/online-evals/cron.log
```

Confirm the job is scheduled:

```bash
crontab -l
```

Then check in Langfuse that the traces carry `answer_relevancy`, `faithfulness` and `contextual_relevancy` scores.

Watch the log across scheduled runs (`-F` waits for the file if it doesn't exist yet):

```bash
tail -F /opt/campusx-rag/online-evals/cron.log
```

## Troubleshooting

`cron.log` doesn't exist: it is created when the first cron tick (or manual run) executes. Wait for the next quarter-hour, then check:

```bash
systemctl status cron --no-pager
```

```bash
grep CRON /var/log/syslog | tail -20
```

If `/var/log/syslog` is missing, use `journalctl -u cron --since "30 min ago" --no-pager`.

```bash
ls -l /usr/bin/flock
```

## Updating later

Pull the new script from S3:

```bash
aws s3 cp s3://campusx-rag-online-evals/run_online_eval.py /opt/campusx-rag/online-evals/ --region ap-south-1
```

If a secret changed, re-run the `.env` command from step 2. Leave `eval_checkpoint.json` alone — it records where the last run stopped.

## Notes

- **Ingestion lag:** the checkpoint advances to each run's start time, so a trace Langfuse ingests late (with an earlier start time) can be skipped permanently. If you see gaps, subtract a few minutes of overlap from `since` in the script. Scores use `score_id = {trace_id}_{name}`, so re-scoring does not create duplicates.
- **Long downtime:** after an outage, the next run covers the whole gap and may take a while; `flock` keeps overlapping ticks from piling up.
- **Log growth:** `cron.log` is never rotated; add a logrotate entry if the instance will live long.
