package main

import "time"

// Test-only cadence overrides. The gate machine runs this suite while other
// agents' checks and live prod share the same host and inference fleet, so a
// job's first placement attempt frequently races its instance's spawn. The
// production backoff quantizes every such miss into a 5s rest; shrinking it
// here keeps every assertion identical while removing that contention
// amplifier. Production defaults (see scheduler.go) are untouched.
func init() {
	placementScanBackoff = 100 * time.Millisecond
	autoWakeCheckInterval = 100 * time.Millisecond
}
