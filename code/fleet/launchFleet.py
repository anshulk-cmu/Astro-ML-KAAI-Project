"""Launch the cutout-download fleet (incremental S3 upload + SSH access).

Each worker boots AL2023, installs deps + AWS CLI v2, pulls sample.parquet + fetch_shard.py
from S3, then loops over its assigned block of shards. fetch_shard.py uploads
images_<s>.npy / status_<s>.npy to S3 every CKPT images (continuous storage), so a block or
crash strands at most CKPT images. Workers launch with the anthropic-fellows-key pair (+ the
already-open SSH rule) so we can SSH in: `ssh -i <key> ec2-user@<ip>` then `sudo tail /root/shard_<s>.log`.

Usage:
  python launch_fleet.py <N_WORKERS> <TOTAL_SHARDS> <ondemand|spot> [CKPT]
  python launch_fleet.py canary            # 1 worker, tiny shard 0 of TOTAL=10000

  python launch_fleet.py 12 48 ondemand 250    # the real run: 12 IPs, 48 shards
"""
import os
import sys
import subprocess
import tempfile

REGION = "us-east-1"
BUCKET = "aion-proto-333650975919"
AMI = "ami-00e801948462f718a"
ITYPE = "t3.small"
PROFILE = "aion-fleet-profile"
SG = "sg-02e6c2a61f1d14fd3"
KEY = "anthropic-fellows-key"
SUBNETS = ["subnet-04aa1afdfba71ea81", "subnet-049caa7dc0a4ff8d6",
           "subnet-0cbd2ff8e3ede8ca2", "subnet-00a1d99de8d54a43e"]
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = open(os.path.join(HERE, "userData.sh"), encoding="utf-8").read()


def userdata(shards, total, ckpt):
    s = TEMPLATE.replace("\r\n", "\n").replace("\r", "\n")
    s = (s.replace("__BUCKET__", BUCKET)
          .replace("__SHARDS__", " ".join(str(x) for x in shards))
          .replace("__TOTAL__", str(total))
          .replace("__CKPT__", str(ckpt)))
    f = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, newline="\n", encoding="utf-8")
    f.write(s)
    f.close()
    return f.name


def launch(idx, shards, total, ckpt, market):
    ud = userdata(shards, total, ckpt)
    cmd = ["aws", "ec2", "run-instances", "--region", REGION,
           "--image-id", AMI, "--instance-type", ITYPE, "--count", "1",
           "--iam-instance-profile", f"Name={PROFILE}", "--key-name", KEY,
           "--subnet-id", SUBNETS[idx % len(SUBNETS)], "--security-group-ids", SG,
           "--instance-initiated-shutdown-behavior", "terminate",
           "--user-data", "file://" + ud.replace("\\", "/"),
           "--tag-specifications",
           f"ResourceType=instance,Tags=[{{Key=Name,Value=aion-worker-{idx}}},{{Key=project,Value=aion-proto}}]",
           "--query", "Instances[0].InstanceId", "--output", "text"]
    if market == "spot":
        cmd += ["--instance-market-options", "MarketType=spot"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(ud)
    if r.returncode == 0:
        return r.stdout.strip(), None
    return None, (r.stderr.strip().splitlines() or ["unknown error"])[-1]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "canary":
        plan, total, ckpt, market = [(0, [0])], 500, 30, "ondemand"
    else:
        n, total = int(sys.argv[1]), int(sys.argv[2])
        market = sys.argv[3] if len(sys.argv) > 3 else "ondemand"
        ckpt = int(sys.argv[4]) if len(sys.argv) > 4 else 250
        plan = [(w, list(range(w * total // n, (w + 1) * total // n))) for w in range(n)]
    print(f"launching {len(plan)} {market} {ITYPE} workers, TOTAL={total} CKPT={ckpt}")
    ids = []
    for idx, shards in plan:
        iid, err = launch(idx, shards, total, ckpt, market)
        if iid:
            print(f"  worker {idx:>2} shards {shards[0]}..{shards[-1]} -> {iid}")
            ids.append(iid)
        else:
            print(f"  worker {idx:>2} shards {shards[0]}..{shards[-1]} -> FAILED: {err}")
    print(f"launched {len(ids)}/{len(plan)}")
    print(" ".join(ids))


if __name__ == "__main__":
    main()
