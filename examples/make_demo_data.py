"""
Generate a small SYNTHETIC subreddit dump (r/demo_homelab) to try the pipeline without
downloading anything. The text is random words from 8 fake topics; it is not real Reddit data.

    python examples/make_demo_data.py
    python organize.py --sub demo_homelab --data-dir examples/demo
"""

import json
import random
from pathlib import Path

random.seed(3)
TOPICS = {
    "docker": "docker compose container networking bridge volumes image",
    "backup": "backup restic offsite borg snapshot retention encryption",
    "virtualization": "proxmox passthrough virtual machine cluster hypervisor iommu",
    "dns": "pihole adguard dns blocking resolver unbound upstream",
    "media": "jellyfin plex transcoding library subtitles streaming",
    "vpn": "wireguard tailscale tunnel remote access firewall port",
    "storage": "truenas raid zfs pool disks scrub parity",
    "power": "electricity power consumption idle watts efficiency bill",
}
COMMON = "server home setup running works issue config".split()

out = Path(__file__).parent / "demo"
out.mkdir(exist_ok=True)
posts, comments, pid, cid = [], [], 0, 0
for words in TOPICS.values():
    vocab = words.split() + COMMON
    say = lambda n: " ".join(random.choice(vocab) for _ in range(n))
    for _ in range(random.randint(40, 160)):
        pid += 1
        t = 1577836800 + pid * 50000
        posts.append({"id": f"p{pid}", "title": say(6).capitalize(), "selftext": say(40),
                      "author": f"demo_user{pid}", "created_utc": t, "score": random.randint(0, 300),
                      "permalink": f"/r/demo_homelab/comments/p{pid}/"})
        for j in range(random.randint(0, 6)):
            cid += 1
            parent = f"t3_p{pid}" if j == 0 or random.random() < 0.5 else f"t1_c{cid - 1}"
            comments.append({"id": f"c{cid}", "link_id": f"t3_p{pid}", "parent_id": parent,
                             "body": say(25), "author": f"demo_c{cid}", "created_utc": t + j,
                             "score": random.randint(0, 50)})
comments.append({"id": "bot1", "link_id": "t3_p1", "parent_id": "t3_p1", "author": "AutoModerator",
                 "body": "Please read the rules before posting.", "created_utc": 1577836801})

(out / "demo_homelab_posts.jsonl").write_text("\n".join(map(json.dumps, posts)), encoding="utf-8")
(out / "demo_homelab_comments.jsonl").write_text("\n".join(map(json.dumps, comments)), encoding="utf-8")
print(f"{len(posts)} posts and {len(comments)} comments written to {out}")
