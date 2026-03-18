"""Photon integration: crawl a website to discover links and emails.

Photon is an optional dependency, run as a subprocess.
If not installed, this module returns empty results with a warning.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


async def crawl_url(url: str, depth: int = 1) -> dict:
    """Crawl a URL using Photon to discover emails, social links, and URLs.

    Args:
        url: URL to crawl.
        depth: Crawl depth (default 1 — just the target page and its links).

    Returns:
        Dict with keys: emails, social_urls, internal_urls, external_urls.
        Empty dict if Photon is not installed or on error.
    """
    photon_bin = shutil.which("photon")
    if not photon_bin:
        logger.warning(
            "photon not installed — skipping website crawl. "
            "Install with: uv pip install 'profiler[osint]'"
        )
        return {}

    tmpdir = tempfile.mkdtemp(prefix="profiler_photon_")

    try:
        proc = await asyncio.create_subprocess_exec(
            photon_bin,
            "-u",
            url,
            "-l",
            str(depth),
            "-o",
            tmpdir,
            "--keys",  # extract secret keys/tokens
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await asyncio.wait_for(proc.communicate(), timeout=45)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(f"Photon timed out crawling {url}")
            # Still try to read partial results

        # Photon writes results as text files in the output directory
        result = {
            "emails": [],
            "social_urls": [],
            "internal_urls": [],
            "external_urls": [],
        }

        output_dir = Path(tmpdir)

        # Read email file
        email_file = output_dir / "email.txt"
        if email_file.exists():
            result["emails"] = [
                line.strip()
                for line in email_file.read_text().splitlines()
                if line.strip()
            ]

        # Read external URLs
        ext_file = output_dir / "external.txt"
        if ext_file.exists():
            urls = [
                line.strip()
                for line in ext_file.read_text().splitlines()
                if line.strip()
            ]
            # Separate social URLs from generic external URLs
            social_domains = [
                "facebook.com",
                "twitter.com",
                "x.com",
                "instagram.com",
                "linkedin.com",
                "github.com",
                "reddit.com",
                "tiktok.com",
                "youtube.com",
                "medium.com",
                "pinterest.com",
                "t.me",
            ]
            for u in urls:
                if any(d in u for d in social_domains):
                    result["social_urls"].append(u)
                else:
                    result["external_urls"].append(u)

        # Read internal URLs
        int_file = output_dir / "internal.txt"
        if int_file.exists():
            result["internal_urls"] = [
                line.strip()
                for line in int_file.read_text().splitlines()
                if line.strip()
            ]

        total = sum(len(v) for v in result.values())
        logger.info(
            f"Photon: crawled {url} — found {len(result['emails'])} emails, "
            f"{len(result['social_urls'])} social URLs, {total} total items"
        )
        return result

    except Exception as e:
        logger.warning(f"Photon crawl failed for {url}: {e}")
        return {}

    finally:
        # Clean up temp directory
        import shutil as _shutil

        try:
            _shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
