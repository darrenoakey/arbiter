# LTX 2.5 source archive

The runtime builder refuses to run unless
`local/ltx25-runtime-sources/<source.archive>` exists in the canonical
checkout and matches `runtime/ltx25/manifest.json`. That directory is
gitignored. The archive is not stored in git.

`source.archive_sha256` `ee5de2ef67fe98d962d62d5ae59bcd76e9a86e020cd86cdc5803962a9acad27b`
was deleted and could not be recovered. It is not reused. The replacement is a
deterministic repack of the same tree plus the same two overlay files, so
commit, tree, runner, runner-test, patch, and patched-file hashes stay
exactly the values already in the manifest. Only the archive bytes, archive
name, archive sha256, and `release_id` change. The new `release_id` keeps the
previous id and appends `-src` plus the first 12 hex characters of the new
archive sha256. Already-built runtimes keep their old provenance files; they
are not rewritten.

## What the archive contains

- `git archive` of `ltx25-spark` commit
  `46172b97612b9b68e98f862f2b13322f608ab009`
  (tree `cd69b4acf635a012d58fbc1a1fdfbfe4cf337d21`)
- overlay `video_fast_gpu.py`
  `8b350b3fc34fa603dadfc6f2d98f7feab7e72fe2d608412702d50d0b5e413bce`
- overlay `tests/test_video_fast_gpu.py`
  `c11fecea3d3b517fb97b657c4daaf6a9a24d2d36799e8bea8664578a1b41c945`

Those two files are the only permitted dirty files in that checkout. The
packager rejects any other overlay path and any overlay whose sha256 differs.

Tar metadata is normalized: member mtime 0, uid 0, gid 0, empty owner names,
directories `0755`, regular files `0644` or `0755` when the git archive had an
execute bit, relative symlinks kept. Gzip is written with mtime 0 and
compresslevel 9. Two packs of the same inputs on the same Python/zlib are
byte-identical.

## Recreate

On the machine that will run the builder, from this worktree:

```bash
mkdir -p /tmp/ltx25-source/tests
ssh -o BatchMode=yes darren@10.0.0.254 'cd /home/darren/src/ltx25-spark && git rev-parse HEAD && git rev-parse "HEAD^{tree}" && git status --porcelain=v1 && sha256sum video_fast_gpu.py tests/test_video_fast_gpu.py && git archive --format=tar HEAD' > /tmp/ltx25-source/head.tar
scp -o BatchMode=yes \
  darren@10.0.0.254:/home/darren/src/ltx25-spark/video_fast_gpu.py \
  /tmp/ltx25-source/video_fast_gpu.py
scp -o BatchMode=yes \
  darren@10.0.0.254:/home/darren/src/ltx25-spark/tests/test_video_fast_gpu.py \
  /tmp/ltx25-source/tests/test_video_fast_gpu.py
python3 scripts/package_ltx25_source.py pack \
  --git-archive /tmp/ltx25-source/head.tar \
  --overlay video_fast_gpu.py=/tmp/ltx25-source/video_fast_gpu.py \
  --overlay tests/test_video_fast_gpu.py=/tmp/ltx25-source/tests/test_video_fast_gpu.py \
  --expect video_fast_gpu.py=8b350b3fc34fa603dadfc6f2d98f7feab7e72fe2d608412702d50d0b5e413bce \
  --expect tests/test_video_fast_gpu.py=c11fecea3d3b517fb97b657c4daaf6a9a24d2d36799e8bea8664578a1b41c945 \
  --output /tmp/ltx25-source/packed.tar.gz
python3 scripts/package_ltx25_source.py pin \
  --archive /tmp/ltx25-source/packed.tar.gz
```

Stop before `pin` if HEAD, tree, status, or overlay sha256 differ from this
document. `pin` applies `temporal-attention.patch` with the builder's
`--fuzz=0` command and checks every patched-file hash before it changes the
manifest. It then copies the archive to the canonical gitignored directory
`local/ltx25-runtime-sources/`. Do not commit that archive.

`attest` checks an already pinned manifest, including the archive sha256 and
`release_id`. It does not skip or weaken those checks.

`tests/unit/test_ltx25_runtime_builder.py` imports torch for the CPU numerical
check. Run that file detached; do not hold an agent turn open on it.
