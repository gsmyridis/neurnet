from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


def _download_error_message(
    filename: str,
    url: str,
    primary_error: requests.exceptions.RequestException | None,
    backup_url: str | None = None,
    backup_error: requests.exceptions.RequestException | None = None,
) -> str:
    details = [f"Failed to download {filename}."]

    if primary_error is not None:
        details.append(
            f"Primary URL failed ({url}): "
            f"{type(primary_error).__name__}: {primary_error}"
        )

    if backup_url and backup_error is not None:
        details.append(
            f"Backup URL failed ({backup_url}): "
            f"{type(backup_error).__name__}: {backup_error}"
        )

    cert_or_proxy_issue = any(
        isinstance(err, (requests.exceptions.ProxyError, requests.exceptions.SSLError))
        for err in (primary_error, backup_error)
        if err is not None
    )
    if not cert_or_proxy_issue:
        lowered = " ".join(
            str(err).lower() for err in (primary_error, backup_error) if err is not None
        )
        cert_or_proxy_issue = any(
            keyword in lowered for keyword in ("certificate", "ssl", "tls", "proxy")
        )

    if cert_or_proxy_issue:
        details.append(
            "This can happen on work or school machines where a VPN, proxy, or "
            "antivirus tool intercepts HTTPS certificates."
        )

    details.append(
        "See the troubleshooting guide: "
        "https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md "
        "(especially the 'File Download Issues' section)."
    )
    return "\n".join(details)


def download_file(
    url: str,
    out_dir: str | Path = ".",
    backup_url: str | None = None,
) -> Path:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = Path(os.path.join(out_dir_path, filename))

    def try_download(
        u: str,
    ) -> tuple[bool, requests.exceptions.RequestException | None]:
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))

                # Skip download if already complete
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True, None

                # Download in 1 MiB chunks with progress display
                block = 1024 * 1024
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024 * 1024)} MiB / "
                                f"{size_remote // (1024 * 1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True, None
        except requests.exceptions.RequestException as exc:
            return False, exc

    # Try main URL first
    success, primary_error = try_download(url)
    if success:
        return dest

    # Try backup URL if provided
    backup_error = None
    if backup_url:
        print(f"Primary URL ({url}) failed.\nTrying backup URL ({backup_url})...")
        success, backup_error = try_download(backup_url)
        if success:
            return dest

    message = _download_error_message(
        filename=filename,
        url=url,
        primary_error=primary_error,
        backup_url=backup_url,
        backup_error=backup_error,
    )
    raise RuntimeError(message) from (backup_error or primary_error)
