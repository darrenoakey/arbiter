# Release performance

The 2026-10-29 aesthetic ZIP batch release was correct but took 355.492s
queue-to-publish against the 180s target (600s hard ceiling). Its Python leg
alone took 245s; deploy reached health polling at +68s. Binary caching already
worked. The primary measured cause was running every Python product test for
a change to one adapter and its schema.

`./run check-impact --python-tests --base <base> --head <candidate>` shows the
Python plan without executing it. The normal gate uses the same planner.
It follows transitive Python imports, retains registry import smoke and
source-inspection/subprocess tests, and treats isolated runtime construction
and deploy-parser contracts separately. Schema changes retain all statically
reachable schema consumers. Harness/config/runtime changes, missing files,
parse errors, unknown ownership and unsupported paths retain the full Python
contract; failed planning falls back to the previous pytest invocation.
Test-only edits run the changed test files. Integration tests remain excluded
from this local gate because they drive the real Spark fleet.

The real baseline profiling run passed. Its individual slow calls included
immutable runtime construction (69.08s) and photo tile blending (58.27s),
neither of which is an aesthetic ZIP dependency. After independent review
added conservative computed-import/file-loader and source-reader coverage,
the exact historical candidate's selected suite passed in 75.04s wall
(19.15s user CPU, 5.50s system), versus the original 245s gate Python leg.
Eighteen selector regressions include real miniature-repository tests of
indirect imports, source readers, and unrelated-test exclusion.

One queued speedup submission was canceled before admission because a
parallel correctness release already held the serialized gate. A queued
candidate must not spend its release budget behind another full release.
Coordinate ownership and rebase after that release; never submit identical
candidates repeatedly. Timings under concurrent fleet load are measurements,
not guaranteed execution times.

Always start full-suite profiling and `greenline submit` detached, retaining
PID, output and exit status. Park the conversation instead of holding a turn.
For a release inspect the stage/process tree at five minutes; terminate and
reap at ten minutes, diagnose, and change the candidate before resubmitting.
Never extend the ceiling or wait for load to improve.

The bootstrap candidate's full validation subsequently passed in 49s (Python
28s, Go 16s, overlapping real remote tests 17.39s). A prior missing-fixture
failure now fails before expensive validation: both actual Ollama inventories
must contain the chat and embedding models required by the Go integrations.
This preflight does not replace any Go test.

The 70.66s bootstrap attempt then failed safely during deploy: live drain
requests were acknowledged, but the cached dashboard still reported its
pre-drain `draining=false` sample. Production logs confirmed that the only
resume was the abort trap, not an external caller or lease expiry. Such a
sample is not a stop proof. Fresh reads may catch up only before the original
drain deadline, without renewing the lease inside that freshness wait; at the
deadline a false drain still aborts. Malformed responses remain hard failures,
and inconsistent counters retain their three-read bound. Every stop still
requires a validated draining snapshot and the existing active-job rule.
