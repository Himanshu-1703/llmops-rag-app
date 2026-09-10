# Auto-scaled deploy: AWS console setup

Manual, one-time setup for the **ASG + ALB + CodeDeploy** rollout. After this is in
place, every CD Pipeline run (and every scale-out instance) deploys the app
automatically from the revision in [`deploy/codedeploy/`](deploy/codedeploy/).

- **Region:** `ap-south-1`  **Account:** `891377050051`
- Already exists (from the manual deploy): Secrets Manager secret **`api-keys`**,
  ECR repos **`campusx-rag-api`** / **`campusx-rag-frontend`**, and the GitHub
  Actions IAM user behind the `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
  repo secrets.
- **Resource names below are contractual** — [`deploy/codedeploy/`](deploy/codedeploy/)
  and [`.github/workflows/cd.yaml`](.github/workflows/cd.yaml) hard-code them. If
  you change a name, change it in both places.

Architecture:

```
              Internet
                 │
        ┌────────▼─────────┐   :80  → frontend target group (:8501)
        │  campusx-rag-alb │   :8000 → api      target group (:8000)
        └────────┬─────────┘
                 │
        ┌────────▼─────────────────────┐   Launch template installs the
        │ campusx-rag-asg  (1 → 3)     │   CodeDeploy agent; CodeDeploy runs
        │  Ubuntu 24.04 · t3.large     │   deploy/codedeploy/ hooks on each box
        └──────────────────────────────┘
                 ▲
     CD Pipeline → S3 revision → CodeDeploy (campusx-rag / campusx-rag-dg)
```

---

## 1. S3 bucket for CodeDeploy revisions

**S3 → Create bucket**

- Name: **`campusx-rag-codedeploy`**
- Region: `ap-south-1`
- Block *all* public access: **on**
- Default encryption: SSE-S3 (default)

---

## 2. IAM roles

### 2a. EC2 instance role — `campusx-rag-instance-role`

**IAM → Roles → Create role → AWS service → EC2**

Attach:

- `AmazonEC2ContainerRegistryReadOnly` — pull images from ECR

Add an inline policy **`campusx-rag-instance-inline`**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadApiKeysSecret",
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:ap-south-1:891377050051:secret:api-keys-*"
    },
    {
      "Sid": "ReadCodeDeployBundles",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::campusx-rag-codedeploy",
        "arn:aws:s3:::campusx-rag-codedeploy/*"
      ]
    }
  ]
}
```

Name the role **`campusx-rag-instance-role`**. (The console creates the matching
instance profile of the same name.)

### 2b. CodeDeploy service role — `campusx-rag-codedeploy-role`

**IAM → Roles → Create role → AWS service → CodeDeploy → "CodeDeploy" use case**

- Attach the AWS managed policy **`AWSCodeDeployRole`** (added automatically for
  that use case).
- Name: **`campusx-rag-codedeploy-role`**.

### 2c. Extend the GitHub Actions IAM user

The user already has ECR push rights. Add an inline policy
**`campusx-rag-cd-deploy`**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PutRevision",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::campusx-rag-codedeploy/*"
    },
    {
      "Sid": "CreateDeployments",
      "Effect": "Allow",
      "Action": [
        "codedeploy:CreateDeployment",
        "codedeploy:GetDeployment",
        "codedeploy:GetDeploymentConfig",
        "codedeploy:RegisterApplicationRevision",
        "codedeploy:GetApplicationRevision"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## 3. Security groups

**VPC → Security groups → Create** (use the default VPC unless you have another).

### `campusx-rag-alb-sg` (load balancer)

| Direction | Type | Port | Source |
| --- | --- | --- | --- |
| Inbound | Custom TCP | 80 | `0.0.0.0/0` |
| Inbound | Custom TCP | 8000 | `0.0.0.0/0` |
| Outbound | All traffic | — | `0.0.0.0/0` |

### `campusx-rag-instance-sg` (EC2 instances)

| Direction | Type | Port | Source |
| --- | --- | --- | --- |
| Inbound | Custom TCP | 8501 | `campusx-rag-alb-sg` |
| Inbound | Custom TCP | 8000 | `campusx-rag-alb-sg` |
| Inbound | SSH | 22 | *your IP only* (optional) |
| Outbound | All traffic | — | `0.0.0.0/0` |

> ⚠️ The `:8000` ALB listener is world-open, and `/chat` is unauthenticated and
> spends OpenAI credits. Before any real use, narrow `campusx-rag-alb-sg` :8000
> to known IPs, or put auth in front.

---

## 4. Target groups

**EC2 → Target groups → Create target group** (type **Instances**, same VPC,
protocol **HTTP**).

### `campusx-rag-frontend-tg`

- Port **8501**
- Health check path **`/_stcore/health`**
- Advanced: healthy threshold 2, unhealthy 3, timeout 5s, interval 30s, success code 200

### `campusx-rag-api-tg`

- Port **8000**
- Health check path **`/health`**
- Same advanced settings

Do **not** register targets by hand — the ASG does it in step 7.

---

## 5. Application Load Balancer

**EC2 → Load balancers → Create → Application Load Balancer**

- Name: **`campusx-rag-alb`**
- Scheme: **Internet-facing**, IP type IPv4
- Network: the VPC, **2+ public subnets** (different AZs)
- Security group: **`campusx-rag-alb-sg`** (remove the default)
- Listeners:
  - **HTTP : 80** → forward to **`campusx-rag-frontend-tg`**
  - After creation: **Add listener → HTTP : 8000** → forward to **`campusx-rag-api-tg`**

Note the ALB's **DNS name** for later.

---

## 6. Launch template

**EC2 → Launch templates → Create launch template**

- Name: **`campusx-rag-lt`**
- AMI: **Ubuntu Server 24.04 LTS**, architecture **64-bit (x86)**
- Instance type: **t3.large** (the API loads the baked Chroma store at startup;
  t3.medium / 4 GB is the practical floor)
- Key pair: your key (for SSH debugging)
- Network settings → **Security groups**: **`campusx-rag-instance-sg`**
  (leave subnet "Don't include in template" — the ASG chooses)
- Advanced details:
  - **IAM instance profile**: **`campusx-rag-instance-role`**
  - **Storage**: 1 volume, **gp3**, **30 GiB**
  - **User data** (installs the CodeDeploy agent — latest v2, verified against
    the AWS docs for `ap-south-1`):

```bash
#!/bin/bash
set -euxo pipefail
apt update
apt install -y wget
cd /home/ubuntu
wget https://aws-codedeploy-ap-south-1.s3.ap-south-1.amazonaws.com/latestv2/install
chmod +x ./install
./install auto
systemctl enable --now codedeploy-agent
systemctl status codedeploy-agent --no-pager
```

> Docker Engine and AWS CLI v2 are **not** installed here — the CodeDeploy
> `BeforeInstall` hook ([`deploy/codedeploy/scripts/before_install.sh`](deploy/codedeploy/scripts/before_install.sh))
> installs them on the first deploy and skips on later ones. If scale-out speed
> becomes a problem, move those installs into this user-data or bake a custom AMI.

---

## 7. Auto Scaling Group

**EC2 → Auto Scaling groups → Create Auto Scaling group**

- Name: **`campusx-rag-asg`**
- Launch template: **`campusx-rag-lt`**, version **Latest**
- Network: the VPC, the **same 2+ public subnets** as the ALB
- **Attach to an existing load balancer → Choose target groups**:
  select **both** `campusx-rag-frontend-tg` and `campusx-rag-api-tg`
- **Health checks**: turn on **Elastic Load Balancing** health checks (EC2 stays on);
  **health check grace period 600s** (cold start + the 120s ValidateService wait)
- Group size: **desired 1, minimum 1, maximum 3**
- **Automatic scaling → Target tracking policy**:
  - Metric type: **Average CPU utilization**
  - Target value: **50**
  - Instance warmup: **300** seconds

---

## 8. CodeDeploy

### 8a. Application

**CodeDeploy → Applications → Create application**

- Name: **`campusx-rag`**
- Compute platform: **EC2/On-premises**

### 8b. Deployment group

**Create deployment group**

- Name: **`campusx-rag-dg`**
- Service role: **`campusx-rag-codedeploy-role`**
- Deployment type: **In-place**
- Environment configuration: **Amazon EC2 Auto Scaling groups** → **`campusx-rag-asg`**
- Deployment settings: **`CodeDeployDefault.OneAtATime`**
- **Load balancer**: enable "Enable load balancing", select **both** target groups
  (`campusx-rag-frontend-tg`, `campusx-rag-api-tg`)
- **Advanced → Rollbacks**: "Roll back when a deployment fails" **on**

Attaching the ASG makes CodeDeploy install its own ASG lifecycle hook — so any
instance the ASG launches later automatically gets the last successful revision
before it goes into service.

---

## 9. First deployment (bootstrap)

The ASG's initial instance boots with the agent but no app yet. Kick the first
rollout:

1. **GitHub → Actions → CD Pipeline → Run workflow** (`workflow_dispatch`).
2. It builds/pushes images, then the **`deploy`** job uploads the revision and
   calls `aws deploy create-deployment`.
3. Watch **CodeDeploy → Deployments →** the running deployment → the instance's
   **lifecycle events**. First run takes ~6–9 min (Docker + CLI install +
   the 120s health-check wait).
   
---

## Verify

```bash
curl http://<alb-dns-name>/                        # Streamlit UI HTML
curl http://<alb-dns-name>:8000/health             # {"message":"welcome to campusx chatbot"}
curl http://<alb-dns-name>:8000/health/dependencies # {"llm_health":"healthy","retriever_health":"healthy","overall_health":"healthy"}
```

- **EC2 → Target groups → `campusx-rag-frontend-tg` / `campusx-rag-api-tg`** →
  Targets tab: the instance shows **healthy** in both.
- Open `http://<alb-dns-name>/` in a browser and ask a question.

Once green, terminate the old hand-provisioned instance.

### Scaling test

```bash
ssh ubuntu@<instance-ip>
sudo apt-get install -y stress-ng && stress-ng --cpu 0 --timeout 600s
```

**EC2 → Auto Scaling groups → `campusx-rag-asg` → Activity** shows a scale-out
within ~3–5 min; the new instance runs a CodeDeploy deployment automatically and
joins both target groups. Stop the load → it scales back to 1 (default 300s
cooldown / 15-min alarm).


---

## After bootstrap: normal operation

Nothing manual. When CI promotes a challenger, **CD Pipeline** runs on its own:
build → push images → assemble revision → `create-deployment` → wait. `.env` is
rebuilt from Secrets Manager on every deploy, so rotating the `api-keys` secret
just needs a re-run of CD (Actions → CD Pipeline → Run workflow).
