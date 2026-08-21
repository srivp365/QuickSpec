# main/setup_model.py
import zipfile
import httpx
from main.paths import MODEL_PATH, RERANKER_PATH
from main.display import make_progress

MODEL_DOWNLOAD_URL = "https://github.com/srivp365/QuickSpec/releases/download/onnx_model/model.zip"

def ensure_model_ready():
    embed_check = MODEL_PATH / "onnx" / "model_qint8_avx512_vnni.onnx"
    if embed_check.exists() and any(RERANKER_PATH.glob("*.onnx")):
        return

    print("First run: downloading models (this happens once)...")
    dest = MODEL_PATH.parent  # APP_DATA / "model"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "model.zip"

    with httpx.stream("GET", MODEL_DOWNLOAD_URL, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with make_progress() as progress, open(zip_path, "wb") as f:
            task = progress.add_task("Downloading model...", total=total)
            for chunk in response.iter_bytes():
                f.write(chunk)
                progress.update(task, advance=len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)  # zip's internal onnx/ and reranker_onnx/ folders land under dest correctly

    zip_path.unlink()
    print("Models ready.")
