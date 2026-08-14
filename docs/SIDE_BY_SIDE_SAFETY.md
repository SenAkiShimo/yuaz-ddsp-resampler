# Side-by-side safety — v0.2.8ai.14

v0.2.8ai.14 is installed alongside v0.2.8ai.13 rather than replacing it.

## Separate runtime resources

The two releases use independent runtime resources:

```text
v0.2.8ai.13 runtime:  ~/Library/Application Support/YuazDDSP/0.2.8ai.13
v0.2.8ai.14 runtime:  ~/Library/Application Support/YuazDDSP/0.2.8ai.14

v0.2.8ai.13 port:     47885
v0.2.8ai.14 port:     47886
```

Each release has its own OpenUtau wrapper and manifest.

## Separate voicebank state

v0.2.8ai.14 writes only to:

```text
.yuaz-0.2.8ai14/
```

Existing `.yuaz-0.2.8ai13/` state is not used as ai.14 learned-state fallback and is not renamed, migrated, overwritten, or deleted.

Deep-trained files and derived caches use ai.14-specific names to prevent collisions with older state layouts.

## Deep preservation snapshot

Before an ai.14 Deep operation begins, the preservation step can snapshot the installed ai.13 runtime/wrapper and available ai.13 voicebank state into:

```text
~/Documents/Yuaz-DDSP-Backups/ai14-preservation/
```

Rebuildable caches are excluded from this safety snapshot.

## Purge policy

`purge-previous-version.command` is disabled in v0.2.8ai.14. Installation and configuration do not invoke predecessor purge or state migration.

Git history preserves the v0.2.8ai.13 source revision. Distribution archives may additionally bundle a read-only source snapshot for regression comparison; such snapshots are never used as writable runtime state.
