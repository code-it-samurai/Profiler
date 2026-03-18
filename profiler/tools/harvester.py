"""theHarvester integration: discover emails, URLs, and hosts for a domain.

theHarvester is a system tool (not a pip package). If not installed,
this module returns empty results with a warning.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


async def harvest(domain_or_name: str) -> dict:
    """Run theHarvester to discover emails, URLs, and hosts.

    Args:
        domain_or_name: Domain name or search term to harvest.

    Returns:
        Dict with keys: emails, urls, hosts.
        Empty dict if theHarvester is not installed or on error.
    """
    harvester_bin = shutil.which("theHarvester")
    if not harvester_bin:
        logger.warning(
            "theHarvester not installed — skipping harvest. "
            "See: https://github.com/laramies/theHarvester"
        )
        return {}

    tmpfile = tempfile.mktemp(prefix="profiler_harvest_", suffix=".json")

    try:
        proc = await asyncio.create_subprocess_exec(
            harvester_bin,
            "-d",
            domain_or_name,
            "-b",
            "duckduckgo,crtsh",
            "-l",
            "100",
            "-f",
            tmpfile,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(f"theHarvester timed out for '{domain_or_name}'")
            # Still try to read partial results

        result = {"emails": [], "urls": [], "hosts": []}

        # theHarvester writes JSON output to the file
        output_path = Path(tmpfile)
        # It may also write as .json (appended by theHarvester)
        json_path = Path(tmpfile + ".json") if not output_path.exists() else output_path
        if not json_path.exists():
            json_path = output_path

        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())

                result["emails"] = [
                    e for e in data.get("emails", []) if isinstance(e, str) and "@" in e
                ]
                result["urls"] = [
                    u
                    for u in data.get("urls", [])
                    if isinstance(u, str) and u.startswith("http")
                ]
                result["hosts"] = [
                    h for h in data.get("hosts", []) if isinstance(h, str)
                ]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse theHarvester output: {e}")

        total = sum(len(v) for v in result.values())
        logger.info(
            f"theHarvester: found {len(result['emails'])} emails, "
            f"{len(result['urls'])} URLs, {len(result['hosts'])} hosts "
            f"for '{domain_or_name}'"
        )
        return result

    except Exception as e:
        logger.warning(f"theHarvester failed for '{domain_or_name}': {e}")
        return {}

    finally:
        # Clean up temp files
        for path in [tmpfile, tmpfile + ".json", tmpfile + ".xml"]:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
