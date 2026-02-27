# Microscopy Thumbnail Viewer

A PyQt6 desktop thumbnail browser optimized for mixed datasets (standard images + large microscopy formats).

## Current capabilities

- Virtualized thumbnail UI (`QListView` + custom delegate)
- Background thumbnail loading (`QThreadPool` + `QRunnable`)
- Two-level cache (RAM + disk cache)
- Search bar and file-type toggles (TIFF, IMS, STK, PSD, PNG, JPG)
- Folder-only left tree pane for clearer directory navigation
- Optional drive scan cache warmup for faster subsequent thumbnail loads
- Spacebar quick-look toggle (open/close)
- Metadata dock (shape, dtype, pixel size when available)
- Context menu actions (reveal path, copy path, clear folder cache)

## Supported file families

- Standard: `.jpg`, `.jpeg`, `.png`, `.webp`, `.psd`, `.ai`
- Microscopy: `.ims`, `.czi`, `.nd2`, `.stk`, `.tif`, `.tiff`, `.ome.tif`, `.ome.tiff`, `.mvd2`, `.acff`

## Installation

Use the included installer script (Windows):

```bat
install.bat
```

The installer creates `.venv`, installs dependencies from `requirements.txt`, and verifies core imports.

## Run

```bat
start.bat
```

## Build (optional)

```bat
.venv\Scripts\python.exe build.py
```

## Notes on large files

- Large non-OME TIFF/STK files use memory-safe fallback paths to avoid full-array allocations.
- IMS files use an HDF5 low-resolution path when available (`h5py`) with BioIO fallback.
- Volocity `.mvd2` and Volocity Library Clipping `.acff` use BioIO + Bio-Formats plugin support.
- Adobe Illustrator `.ai` files use Qt PDF rendering when PDF-compatible, with Pillow fallback for legacy PostScript-style files.
- Folder load state is persisted in the cache database under `~/.microscopy_cache`.
- Drive scan cache warmup currently pre-caches stable formats (`jpg/jpeg/png/webp/psd/ai/ims/stk/tif/tiff`) for startup safety.

## Troubleshooting

- **IMS thumbnails not appearing**
	- Run `install.bat` to ensure `h5py` is installed in `.venv`.
	- Verify with:
		- `.venv\Scripts\python.exe -c "import h5py, bioio; print('ok')"`

- **Console shows BioIO unsupported format messages for TIFF/STK**
	- Non-OME TIFF/STK files are routed through fallback readers first.
	- If a file is malformed or extremely large, a placeholder may be shown instead of crashing.

- **Some Illustrator (`.ai`) files still show as broken**
	- PDF-compatible AI files are supported directly.
	- Older EPS/PostScript-style AI files may require Ghostscript available to Pillow on your system.

- **Volocity files (`.mvd2` / `.acff`) do not open**
	- Re-run `install.bat` to ensure `bioio-bioformats` is installed in `.venv`.
	- Verify with:
		- `.venv\Scripts\python.exe -c "import bioio, bioio_bioformats; print('ok')"`

- **App slows down on huge drive roots**
	- Use file-type toggles and the search box to reduce active thumbnail workload.
	- Open a subfolder instead of the full drive root when possible.
	- Turn on **Enable Drive Scan Cache** to prebuild thumbnail cache in the background for future loads.

- **Preview/metadata appears stale**
	- Right-click an item and use **Clear Cache for Folder**.
	- Cached data is stored under `~/.microscopy_cache`.

- **Environment mismatch after Python upgrade**
	- Re-run `install.bat`; it will recreate `.venv` if Python minor version changed.
