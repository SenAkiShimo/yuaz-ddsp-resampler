# Side-by-side safety — 0.2.8ai.11

| Resource | RC4.2 | 0.2.7 AI.3 | 0.2.8ai | 0.2.8ai.1 | 0.2.8ai.2 | 0.2.8ai.3 | 0.2.8ai.11 |
|---|---|---|---|---|---|---|---|
| Port | 47872 | 47875 | 47876 | 47877 | 47878 | 47879 | 47880 | 47885 |
| Voicebank state | `.yuaz-alpha8-rc3-3` | `.yuaz-alpha8-rc4-3-ai3` | `.yuaz-0.2.8ai` | `.yuaz-0.2.8ai1` | `.yuaz-0.2.8ai2` | `.yuaz-0.2.8ai3` | `.yuaz-0.2.8ai4` | `.yuaz-0.2.8ai11` |

0.2.8ai.11 owns only its final column. Fallback order is own state → 0.2.8ai.3 → 0.2.8ai.2 → 0.2.8ai.1 → 0.2.8ai → AI.3 → RC4.2. Every predecessor is read-only.

The safe purge/uninstall scripts remove only 0.2.8ai.11 resources. Engine snapshots and voicebank Deep backups preserve predecessor resources when present.
