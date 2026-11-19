//go:build !linux

package main

import "regexp"

// killIfOwnedOrphan never kills outside Linux: without /proc and pidfds the
// ownership boundary cannot be established, so the sweep fails closed.
func killIfOwnedOrphan(pid, selfPid int, re *regexp.Regexp) bool {
	return false
}
