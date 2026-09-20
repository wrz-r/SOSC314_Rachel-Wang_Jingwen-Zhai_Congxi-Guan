# Install the tested crawler version and Python dependencies.
from pathlib import Path
import subprocess
import sys

FULL_REPO = Path("/content/weibo-search-sampling")
PINNED_COMMIT = "b4535b71d36ae61d13ab0083a18a152914f14ba7"

if not FULL_REPO.exists():
    subprocess.run(
        ["git", "clone", "--no-checkout",
         "https://github.com/dataabc/weibo-search.git", str(FULL_REPO)],
        check=True,
    )
subprocess.run(
    ["git", "-C", str(FULL_REPO), "fetch", "--depth", "1", "origin", PINNED_COMMIT],
    check=True,
)
subprocess.run(
    ["git", "-C", str(FULL_REPO), "checkout", "--detach", PINNED_COMMIT],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(FULL_REPO / "requirements.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
     "Scrapy==2.12.0", "Twisted==24.11.0", "regex"],
    check=True,
)
print("Repository ready:", FULL_REPO)

