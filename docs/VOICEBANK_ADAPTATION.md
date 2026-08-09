# Voicebank Adaptation

## UTAU-native routing

When a usable `prefix.map` exists, its distinct prefix/suffix groups are the authoritative timbre prototypes. Alias matches are assigned directly. Unmatched entries may be routed into an existing prototype by a known folder association, a pitch note found at the end of a folder name, or observed F0, but these fallback paths cannot create additional prototypes.

Without a usable `prefix.map`, pitch-labelled folders, oto-folder boundaries, and observed F0 remain available as fallback prototype sources.

The normalized `base_alias` removes the matched pitch affix so real recordings of the same phonetic item across different UTAU subbanks can form multipitch training pairs.

At render time, target F0 continuously weights nearby prototypes while the actual input WAV's subbank index acts as a routing hint. This keeps the timbre path aligned with the same sample-selection structure OpenUtau already uses.

## Inspection

Preparation writes `.yuaz/subbanks.json`. `inspect-voicebank.command` prints a routing summary first, including prototype count, labels, whether `prefix.map` was authoritative, how many entries used fallback assignment, and whether any fallback-created prototypes exist.

For a prefix-mapped bank, `fallback_created_prototypes` should be `0`.

## Articulation metadata

Cache format 5 stores per-oto articulation boundaries derived from that entry's own offset/cutoff/fixed-region values. This matters for CVVC/VCV banks where many aliases may point into the same WAV but represent different usable slices.

Compatible older caches are upgraded in place. The existing Yuaz latent, detail features and anti-leak perturbation latents remain reusable.
