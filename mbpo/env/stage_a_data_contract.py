"""Read-only NSRDB manifest + annual-file provenance for Stage-A checkpoints.

Annual file contents are hashed once per Python process; no weather file is
rewritten, resampled, downloaded, or substituted. The cache is keyed by file
metadata and each run compares the resulting hashes to checkpoint physics.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _digest_cached(path,size,mtime_ns):
    return sha_file(path)


def nsrdb_dataset_fingerprint(manifest_path):
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    source=Path(manifest_path).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    manifest=load_scenario_manifest(str(source))
    years={}
    for row in manifest['scenarios']:
        year=int(row.get('source_year',row['year']))
        path=Path(row['_year_path']).resolve()
        if year in years and years[year]!=path:
            raise ValueError(f'Conflicting NSRDB paths for year {year}')
        years[year]=path
    records={}
    for year,path in sorted(years.items()):
        if not path.is_file():raise FileNotFoundError(path)
        stat=path.stat()
        records[str(year)]=_digest_cached(str(path),stat.st_size,stat.st_mtime_ns)
    return {'nsrdb_manifest_sha256':sha_file(source),
            'nsrdb_year_file_sha256':records,
            'nsrdb_year_count':len(records),
            'nsrdb_source_contract':'local_csv_utc_native_5min_no_data_synthesis'}
