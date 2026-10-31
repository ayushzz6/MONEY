#!/bin/bash
# Set up MONEYV2 on a fresh Ubuntu server (Oracle Cloud, user "ubuntu"). Safe to run again to update.
#   bash setup.sh            then copy .env to /home/ubuntu/MONEY/.env and: sudo systemctl restart moneyv2
set -euo pipefail
REPO=https://github.com/ayushzz6/MONEY.git
DIR=/home/ubuntu/MONEY

sudo timedatectl set-timezone Asia/Kolkata
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q git python3-venv python3-pip

# 2 GB of swap: pandas and the 60-day history are heavy on a small server
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q /swapfile /etc/fstab || echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone "$REPO" "$DIR"; fi
cd "$DIR"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements-server.txt
chmod +x run.sh

sudo cp deploy/moneyv2.service deploy/moneyv2-morning.service deploy/moneyv2-morning.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now moneyv2-morning.timer
if [ -f .env ]; then
  chmod 600 .env
  sudo systemctl enable moneyv2 && sudo systemctl restart moneyv2
  echo "MONEYV2 started. Logs: journalctl -u moneyv2 -f"
else
  sudo systemctl enable moneyv2
  echo "Copy your .env to $DIR/.env, then: sudo systemctl restart moneyv2"
fi
