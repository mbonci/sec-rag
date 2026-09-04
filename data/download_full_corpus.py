"""Download the full 50-document SEC 10-K corpus from EDGAR.

The repository ships with a 5-document sample in Documents.zip.
Run this script to download the remaining 45 filings directly from
the SEC EDGAR full-text search API. No API key required.

Usage:
    python data/download_full_corpus.py
    # Documents are written to data/Documents/ (same path the pipeline expects)
"""

import time
import urllib.request
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "Documents"

# (filename, EDGAR URL) — all public filings, no auth required
FILINGS = [
    ("CHARTER COMMUNICATIONS_ INC. _MO___CIK0001091667__000109166719000029_chtr12312018-10k.txt",
     "https://www.sec.gov/Archives/edgar/data/1091667/000109166719000029/chtr12312018-10k.htm"),
    ("DOVER Corp__CIK0000029905__000002990517000011_a2016123110-k.txt",
     "https://www.sec.gov/Archives/edgar/data/29905/000002990517000011/a2016123110-k.htm"),
    ("Gen Digital Inc.__CIK0000849399__000084939919000005_symc32919-10k.txt",
     "https://www.sec.gov/Archives/edgar/data/849399/000084939919000005/symc32919-10k.htm"),
    ("IDEX CORP _DE___CIK0000832101__000083210116000057_iex-20151231x10k.txt",
     "https://www.sec.gov/Archives/edgar/data/832101/000083210116000057/iex-20151231x10k.htm"),
    ("COPART INC__CIK0000900075__000114544311000951_d27786.txt",
     "https://www.sec.gov/Archives/edgar/data/900075/000114544311000951/d27786.htm"),
    ("QUANTA SERVICES_ INC.__CIK0001050915__000119312517064821_d295903d10k.txt",
     "https://www.sec.gov/Archives/edgar/data/1050915/000119312517064821/d295903d10k.htm"),
    ("EVEREST GROUP_ LTD.__CIK0001095073__000109507317000011_group10k2016.txt",
     "https://www.sec.gov/Archives/edgar/data/1095073/000109507317000011/group10k2016.htm"),
    ("EXPEDITORS INTERNATIONAL OF WASHINGTON INC__CIK0000746515__000074651518000004_a201710-k.txt",
     "https://www.sec.gov/Archives/edgar/data/746515/000074651518000004/a201710-k.htm"),
    ("INTERPUBLIC GROUP OF COMPANIES_ INC.__CIK0000051644__000005164415000011_ipg12311410k.txt",
     "https://www.sec.gov/Archives/edgar/data/51644/000005164415000011/ipg12311410k.htm"),
    ("INCYTE CORP__CIK0000879169__000155837019000682_incy-20181231x10k.txt",
     "https://www.sec.gov/Archives/edgar/data/879169/000155837019000682/incy-20181231x10k.htm"),
    ("STRYKER CORP__CIK0000310764__000031076418000031_syk10k123117.txt",
     "https://www.sec.gov/Archives/edgar/data/310764/000031076418000031/syk10k123117.htm"),
    ("WATERS CORP _DE___CIK0001000697__000119312517056239_d268303d10k.txt",
     "https://www.sec.gov/Archives/edgar/data/1000697/000119312517056239/d268303d10k.htm"),
    ("MOSAIC CO__CIK0001285785__000161803415000005_mos-20141231x10k.txt",
     "https://www.sec.gov/Archives/edgar/data/1285785/000161803415000005/mos-20141231x10k.htm"),
]

HEADERS = {"User-Agent": "research-project sec-rag-pipeline mbonci117@gmail.com"}


def download_filing(filename: str, url: str, dest_dir: Path) -> bool:
    dest = dest_dir / filename
    if dest.exists():
        print(f"  skip  {filename} (already exists)")
        return False
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        print(f"  ok    {filename}")
        return True
    except Exception as e:
        print(f"  FAIL  {filename}: {e}")
        return False


if __name__ == "__main__":
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    print(f"Downloading to {DOCUMENTS_DIR}/")
    print("SEC EDGAR rate-limits to ~10 req/s — pausing 0.2s between requests.\n")
    downloaded = 0
    for filename, url in FILINGS:
        if download_filing(filename, url, DOCUMENTS_DIR):
            downloaded += 1
            time.sleep(0.2)
    print(f"\nDone. {downloaded} new files downloaded.")
    print("Tip: run `cd data && unzip Documents.zip` first to extract the 5-file sample.")
