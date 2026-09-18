"""Download the data: Financial PhraseBank (Malo, Sinha et al., 2014).

About 4,800 sentences from financial news, each labelled positive, negative or
neutral - for an investor, is this good or bad news for the share price? -
by 5 to 8 annotators with a finance background. Licence CC BY-NC-SA 3.0
(non-commercial), from the Hugging Face copy of the original release. Checked
against a known SHA-256.

    python get_data.py
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/"
       "data/FinancialPhraseBank-v1.0.zip")
SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"
RAW = Path(__file__).parent / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "phrasebank.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch - got {digest}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    print(f"ok  {RAW / 'FinancialPhraseBank-v1.0'}")


if __name__ == "__main__":
    main()
