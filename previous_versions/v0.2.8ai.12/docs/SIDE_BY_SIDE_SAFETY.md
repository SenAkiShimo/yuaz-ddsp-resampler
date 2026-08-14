# Side-by-side / migration safety — 0.2.8ai.12

0.2.8ai.12 writes only `.yuaz-0.2.8ai12` and `voicebank_registry-0.2.8ai12.json` before migration is committed.

Preferred read-only migration order starts with `.yuaz-0.2.8ai11`, then ai.10 and older namespaces. A new ai.12 generation is built in staging, critical state is validated, and only then is it committed ACTIVE.

The installer creates a pre-ai.12 snapshot before removing old installed runtimes/wrappers/state containers. Source WAV/OTO, datasets and `~/Documents/Yuaz-DDSP-Backups` are not deleted.

The **source code of ai.11 is not discarded**: the exact previous package is bundled under `previous_versions/v0.2.8ai.11/`, and ai.11 synthesis/mixer functions remain callable as the compatibility fallback in the active ai.12 core.
