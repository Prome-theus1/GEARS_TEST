#!/usr/bin/env python3
import os
import subprocess

path = "dixit"
prefix = "offline-run-"

for dp, ds, _ in os.walk(path):
    for d in ds:
        if d.startswith(prefix):
            run_dir = os.path.join(dp, d)
            print("sync:", run_dir)
            subprocess.run(["wandb", "sync", run_dir], check=False)