#!/usr/bin/env python3
"""Bounded, read-only check of the existing MLflow HTTP service; never migrates a DB."""

import argparse
import importlib.metadata
import json
import os
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = parser.parse_args(argv)
    uri = args.tracking_uri
    parts = urlsplit(uri or "")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        parser.error("set an explicit HTTP(S) MLflow endpoint")
    if parts.username or parts.password or parts.query or parts.fragment:
        parser.error("keep credentials in standard MLflow environment variables, outside the URI")
    os.environ.update(MLFLOW_HTTP_REQUEST_TIMEOUT="5", MLFLOW_HTTP_REQUEST_MAX_RETRIES="0",
                      MLFLOW_DISABLE_AGENT_HINT="1")
    try:
        with urlopen(uri.rstrip("/") + "/health", timeout=5) as response:
            if response.read(1024).strip() != b"OK":
                raise ValueError("unexpected health response")
        from mlflow import MlflowClient
        experiments = MlflowClient(tracking_uri=uri).search_experiments(max_results=1)
    except Exception as error:
        # Client exceptions may contain an authenticated URL; print only the type.
        print(f"doctor: MLflow read-only check failed ({type(error).__name__})", file=sys.stderr)
        return 1
    print(json.dumps({"health": "OK", "client_version": importlib.metadata.version("mlflow-skinny"),
                      "experiment_query": "OK", "experiments_returned": len(experiments),
                      "writes": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
