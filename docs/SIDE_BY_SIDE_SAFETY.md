# Side-by-side / migration safety — 0.2.8ai.13

0.2.8ai.13 writes new prepared state only into `.yuaz-0.2.8ai13` and uses `voicebank_registry-0.2.8ai13.json`.

The preferred read-only migration source is `.yuaz-0.2.8ai12`, followed by ai.11 and older namespaces. A new ai.13 generation is built in staging, its critical state is validated, and only then is it committed ACTIVE.

Before installed predecessors are purged, `backup-current-stable.command` snapshots the current ai.12 runtime/wrapper and available voicebank state. Source WAV/OTO, datasets and `~/Documents/Yuaz-DDSP-Backups` are not deleted.

The **source code of ai.12 is not discarded**. The exact previous package is bundled at `previous_versions/v0.2.8ai.12/`; that snapshot also contains ai.12's preserved ai.11 tree. In the active ai.13 source, the ai.12 upper-band head/mixer and ai.11 synthesis fallback remain callable. ai.13 adds a new route instead of deleting those implementations.

After all state migrations validate, the installer may remove older *installed* Yuaz runtimes/wrappers and migrated state containers so OpenUtau does not keep multiple active Yuaz versions. That cleanup does not remove the preserved source snapshots inside the ai.13 package or the pre-purge backups.
