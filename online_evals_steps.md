# Online Evals on EC2 — Setup Steps

Deploys `src/evals/online_evals/run_online_eval.py` to a separate EC2 instance and runs it as a cron job every 15 minutes. Each run scores new Langfuse traces (`answer_relevancy`, `faithfulness`, `contextual_relevancy`) and pushes the scores back to Langfuse.

**Layout on the instance**

```
/opt/campusx-rag/
├── .env                               # created from Secrets Manager (chmod 600)
├── .uv-cache/                         # uv package cache (safe to delete)
├── bin/uv                             # uv, installs Python 3.12 and the libraries
├── python/                            # Python 3.12, downloaded by uv
├── venv/                              # Python 3.12 virtualenv
├── run_eval.sh                        # wrapper called by cron (cd + flock + log)
└── online-evals/
    ├── run_online_eval.py             # pulled from S3
    ├── requirements-online-evals.txt  # pulled from S3
    ├── eval_checkpoint.json           # created on first run; do not delete
    └── cron.log                       # script output
```

## Prerequisites

- EC2 instance running Ubuntu (t3.small, ~15 GiB disk is enough). Only port 22 needs to be open — nothing listens.
- IAM instance profile attached to the instance with:
  - `secretsmanager:GetSecretValue` on `api-keys-*`
  - `s3:GetObject` and `s3:ListBucket` on `campusx-rag-online-evals`
- Secret `api-keys` (region `ap-south-1`) containing `OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL`.
- S3 bucket `campusx-rag-online-evals`.

## 1. Local machine: upload to S3

The script calls `client.api.observations.get_many(...)`, which depends on the Langfuse SDK version, so pin the same versions used locally. Run these from the repo root.

```bash
uv pip freeze | grep -E '^(deepeval|langfuse|python-dotenv)==' > requirements-online-evals.txt
```

It should list three pinned (`==`) lines:

```bash
cat requirements-online-evals.txt
```

```bash
aws s3 cp requirements-online-evals.txt s3://campusx-rag-online-evals/ --region ap-south-1
```

```bash
aws s3 cp src/evals/online_evals/run_online_eval.py s3://campusx-rag-online-evals/ --region ap-south-1
```

## 2. EC2 (Ubuntu): one-time setup

For a fresh instance. Log in as your normal user (e.g. `ubuntu`), paste one block at a time, and wait for it to finish before the next. Don't add `sudo` where it isn't shown: a folder or venv created by root can't be written to later, which is what causes `Permission denied` on the venv.

Install packages (`NEEDRESTART_MODE=a` stops Ubuntu's "restart services?" dialog from swallowing what you paste next):

```bash
sudo apt-get update
sudo NEEDRESTART_MODE=a apt-get install -y jq unzip cron
```

Install the AWS CLI (skip if `aws --version` already works). The last line deletes the installer files, which take a few hundred MB of `/tmp`; leaving them there can fill `/tmp` and make later installs fail with `Disk quota exceeded`:

```bash
curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
sudo /tmp/aws/install
rm -rf /tmp/aws /tmp/awscliv2.zip
```

Check the instance's IAM role is attached (should print an ARN):

```bash
aws sts get-caller-identity
```

Create the folders, owned by you:

```bash
sudo mkdir -p /opt/campusx-rag/online-evals
sudo chown -R "$(id -un):$(id -gn)" /opt/campusx-rag
```

The owner must be your login user (e.g. `ubuntu`), not `root`. If it says `root`, the venv step below fails with `Permission denied`:

```bash
ls -ld /opt/campusx-rag
```

Pull the script and requirements from S3:

```bash
aws s3 cp s3://campusx-rag-online-evals/run_online_eval.py /opt/campusx-rag/online-evals/ --region ap-south-1
aws s3 cp s3://campusx-rag-online-evals/requirements-online-evals.txt /opt/campusx-rag/online-evals/ --region ap-south-1
```

Install [uv](https://docs.astral.sh/uv/) into `/opt/campusx-rag/bin`. Ubuntu's own Python is newer than the 3.12 you develop on, and uv installs 3.12 without a PPA. uv, Python and the package cache all stay under `/opt/campusx-rag`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/opt/campusx-rag/bin sh
```

Create the venv with Python 3.12. uv downloads it the first time; `--clear` replaces any earlier venv:

```bash
UV_PYTHON_INSTALL_DIR=/opt/campusx-rag/python /opt/campusx-rag/bin/uv venv --clear --python 3.12 /opt/campusx-rag/venv
```

Install the libraries. It ends with a list of `+ package==version` lines:

```bash
UV_CACHE_DIR=/opt/campusx-rag/.uv-cache /opt/campusx-rag/bin/uv pip install --python /opt/campusx-rag/venv/bin/python -r /opt/campusx-rag/online-evals/requirements-online-evals.txt
```

Check the versions. Expect `Python 3.12.x`, then the three pinned lines (`deepeval==4.0.7`, `langfuse==4.14.1`, `python-dotenv==1.2.2`). The venv has no `pip`; use `uv pip` for anything package-related:

```bash
/opt/campusx-rag/venv/bin/python --version
```

```bash
/opt/campusx-rag/bin/uv pip freeze --python /opt/campusx-rag/venv/bin/python | grep -Ei '^(deepeval|langfuse|python-dotenv)=='
```

Write `.env` from Secrets Manager:

```bash
aws secretsmanager get-secret-value --secret-id api-keys --region ap-south-1 \
  --query SecretString --output text \
  | jq -r 'to_entries[] | "\(.key)=\(.value)"' > /opt/campusx-rag/.env
chmod 600 /opt/campusx-rag/.env
```

Create the wrapper script that cron calls:

```bash
cat > /opt/campusx-rag/run_eval.sh <<'EOF'
#!/bin/bash
cd /opt/campusx-rag/online-evals
exec /usr/bin/flock -n /tmp/online_eval.lock /opt/campusx-rag/venv/bin/python run_online_eval.py >> cron.log 2>&1
EOF
chmod +x /opt/campusx-rag/run_eval.sh
```

Schedule it every 15 minutes. This replaces the crontab, which is fine on a fresh instance:

```bash
echo '*/15 * * * * /opt/campusx-rag/run_eval.sh' | crontab -
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

`Permission denied` when creating the venv (`[Errno 13] Permission denied: '/opt/campusx-rag/venv'`): your login user doesn't own `/opt/campusx-rag`, usually because the `chown` never ran or something was created with `sudo`. Take ownership back:

```bash
sudo chown -R "$(id -un):$(id -gn)" /opt/campusx-rag
```

Then re-run the "Create the venv" and "Install the libraries" blocks from step 2 (`--clear` replaces a half-built venv).

`Disk quota exceeded` (`OSError: [Errno 122]`, or `tar: … Cannot write`) while installing: installers unpack into `/tmp`, and on a fresh Ubuntu instance it ran out of room, most likely because the AWS CLI installer files were still in it. Delete them and check the free space:

```bash
rm -rf /tmp/aws /tmp/awscliv2.zip && df -hT /tmp
```
