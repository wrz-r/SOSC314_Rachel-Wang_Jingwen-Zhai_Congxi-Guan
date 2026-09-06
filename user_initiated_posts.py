# Import Path so that we can define and check the folder where the repository will be stored.
from pathlib import Path
# Import subprocess so that we can run Git commands from Python.
import subprocess

# Define the folder where the full-year Weibo Search repository will be stored in Google Colab.
FULL_REPO = Path("/content/weibo-search-2025-full")
# Check whether the repository has already been downloaded.
# If the folder does not exist, clone the repository from GitHub.
if not FULL_REPO.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "--depth", "1",
            "https://github.com/dataabc/weibo-search.git",
            str(FULL_REPO)
        ],
        check=True
    )
# Print the local repository path so that we can confirm where the files are stored.
print("Full-year repository:", FULL_REPO)

# Import getpass so that we can enter the Weibo cookie without displaying
from getpass import getpass
# Ask me to enter their Weibo cookie.
# The cookie is stored in WEIBO_COOKIE, but the entered value remains hidden to protect sensitive account information.
WEIBO_COOKIE = getpass("Paste your Weibo cookie here: ")
# Confirm that the cookie has been stored without revealing its value.
if WEIBO_COOKIE:
    print("Cookie loaded successfully. Its value is hidden.")
else:
    print("No cookie was entered.")


# Install the Python packages required by the Weibo Search repository.
!pip -q install -r /content/weibo-search-2025-full/requirements.txt
# Reinstall the specified versions of Scrapy and Twisted to ensure that they are compatible with the crawler and to avoid errors caused by other installed versions.
!pip -q install --force-reinstall "Scrapy==2.12.0" "Twisted==24.11.0"
# Display the installed Scrapy version and the versions of its main dependencies.
!scrapy version -v










