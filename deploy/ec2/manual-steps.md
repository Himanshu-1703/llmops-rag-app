# EC2 manual deploy
---

## 1. Install Docker Engine

Add Docker's official GPG key:

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

Add the repository to Apt sources:

```bash
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
```

Install the latest version:

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify:

```bash
sudo docker run hello-world
```

---

## 2. Install AWS CLI v2

```bash
sudo apt-get install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Verify:

```bash
aws --version
```

Clean up:

```bash
rm -rf awscliv2.zip aws
```

> On arm64 (Graviton) instances swap the URL for
> `awscli-exe-linux-aarch64.zip`.

---

## 3. Create the app directory

```bash
sudo mkdir -p /opt/campusx-rag
sudo chown "$USER":"$USER" /opt/campusx-rag
cd /opt/campusx-rag
```

The instance's IAM role must already allow `secretsmanager:GetSecretValue`,
ECR pull, and `s3:GetObject` — see [README.md](README.md) step 1. Confirm the
role is attached:

```bash
aws sts get-caller-identity
```

---

## 4. Write `.env` from Secrets Manager (`api-keys`)

```bash
sudo apt-get install jq
aws secretsmanager get-secret-value \
  --secret-id api-keys \
  --region ap-south-1 \
  --query SecretString --output text \
  | jq -r 'to_entries[] | "\(.key)=\(.value)"' > /opt/campusx-rag/.env
chmod 600 /opt/campusx-rag/.env
```

Sanity check (keys only, not values):

```bash
cut -d= -f1 /opt/campusx-rag/.env
```

Expect: `OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_BASE_URL`, `ADMIN_API_KEY`.

---

## 5. Pull the compose file from S3

```bash
aws s3 cp s3://llmops-rag-compose/docker-compose.yml /opt/campusx-rag/docker-compose.yml --region ap-south-1
```

---

## 6. Log in to ECR, pull, and start

```bash
aws ecr get-login-password --region ap-south-1 \
  | sudo docker login --username AWS --password-stdin 891377050051.dkr.ecr.ap-south-1.amazonaws.com
```

```bash
cd /opt/campusx-rag
sudo docker compose pull
sudo docker compose up -d
sudo docker compose ps
```

---

## Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/dependencies   # live LLM + retriever probe, up to ~90s
sudo docker compose logs -f
```

From your laptop: `http://<public-ip>:8501` (frontend), `http://<public-ip>:8000`
(API).

---

## Redeploy later (new `:latest` images or rotated secret)

```bash
cd /opt/campusx-rag
aws secretsmanager get-secret-value --secret-id api-keys --region ap-south-1 \
  --query SecretString --output text \
  | jq -r 'to_entries[] | "\(.key)=\(.value)"' > .env && chmod 600 .env
aws s3 cp s3://llmops-rag-compose/docker-compose.yml ./docker-compose.yml --region ap-south-1
aws ecr get-login-password --region ap-south-1 \
  | sudo docker login --username AWS --password-stdin 891377050051.dkr.ecr.ap-south-1.amazonaws.com
sudo docker compose pull && sudo docker compose up -d
sudo docker image prune -f
```
