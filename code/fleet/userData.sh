#!/bin/bash
# Worker user-data. Placeholders __BUCKET__ __SHARDS__ __TOTAL__ __CKPT__ substituted per worker.
# Each worker loops over its block of shards; fetchShard.py self-uploads images/status to S3
# incrementally (every CKPT images), so a block or crash never strands a whole shard.
set -x
dnf install -y python3-pip unzip >/tmp/dnf.log 2>&1
# Self-contained AWS CLI v2 (the AMI's system aws CLI is missing python deps, e.g. dateutil).
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update >/tmp/awsinstall.log 2>&1
pip3 install -q numpy pandas pyarrow requests astropy
cd /root
/usr/local/bin/aws s3 cp s3://__BUCKET__/sample.parquet sample.parquet
/usr/local/bin/aws s3 cp s3://__BUCKET__/fetchShard.py fetchShard.py
for SHARD in __SHARDS__; do
  SHARD=$SHARD TOTAL=__TOTAL__ BUCKET=__BUCKET__ CKPT=__CKPT__ python3 fetchShard.py > shard_$SHARD.log 2>&1
  /usr/local/bin/aws s3 cp shard_$SHARD.log s3://__BUCKET__/logs/shard_$SHARD.log
done
shutdown -h now
