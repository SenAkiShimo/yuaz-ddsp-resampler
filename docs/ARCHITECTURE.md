# Architecture

```text
UTAU oto slice
│
├─ Region A: original unvoiced consonant / transient
│
├─ Region B: handoff before reliable periodic voicing
│
├─ Region C: early voiced articulation
│       current oto variant -> base_alias
│                    ↓
│       canonical articulation dictionary
│       ├─ real multipitch common trajectory when available
│       └─ timbre-neutral local fallback otherwise
│                    ↓
│       local spectral/formant/energy motion only
│       broad timbre tilt removed + clarity guard
│                    ↓
│       target-F0 DDSP magnitude shaping
│       DDSP phase/excitation remain unique
│
└─ Region D: stable vowel
              ↓
           Yuaz Encoder
              ├─ anti-leak content residual
              ├─ high-frequency detail sidecar
              ├─ global voicebank timbre
              └─ UTAU-native dynamic subbank timbre
                    ↓
                  Yuaz DDSP
                    ↓
             constrained Fidelity Refiner
                    ↓
           complete hybrid waveform
                    ↓
       strict final-render active-RMS normalize
                    ↓
          local soft peak guard only if needed
                    ↓
             OpenUtau note volume
```

The renderer keeps one periodic source after reliable voicing begins. Canonical articulation changes magnitude-envelope motion only; it does not mix a second voiced waveform into DDSP.

For shared CVVC/VCV WAVs, the registry resolves the current oto variant by `offset / consonant / cutoff` before selecting `base_alias`, subbank routing, and canonical articulation.
