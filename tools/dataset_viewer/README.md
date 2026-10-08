# EUVP Underwater ImageNet sample viewer

A local, read-only browser for choosing three verified `trainA`/`trainB` image pairs. It reads the existing originals directly from `data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet/`; it does not copy, edit, or transform dataset files.

## Run locally

From the project root, start the server with Python 3:

```powershell
.\.venv\Scripts\python.exe tools\dataset_viewer\server.py --host 127.0.0.1 --port 8765
```

Run this command from the project root, keep the terminal open, and open <http://127.0.0.1:8765> in a browser. Stop the server with `Ctrl+C`. The viewer binds to localhost and needs no additional packages.

## Use

- The initial view shuffles the verified matching filenames and shows 16 pairs. Choose 12 or 20 from the selector.
- Search matches filenames; Previous/Next move through results, while Random pair jumps to a random result.
- Select exactly three pairs. “Show representative candidates” ranks up to 320 input images with simple brightness, contrast, and color statistics, then suggests three visually varied candidates. It is only a heuristic; inspect and choose the final samples yourself.
- Download Figure PNG exports a 2400 px wide, white-background, bordered figure from the original image pixels. Each image is fitted without distortion. Copy filenames copies the `trainA/…` and `trainB/…` paths for the selected pairs.
- A filename appears only when it exists in both folders. The status line reports unmatched files, if present.

The browser requires JavaScript and stays on your machine; no external services or dependencies are used.
