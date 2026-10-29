# Whistle (and Needle 2) on the Waveshare ESP32-P4-ETH

Port of the [whistle-esp32](../original/whistle-esp32) voice-to-text demo
(and, next, [needle-2-esp32](../original/needle-2-esp32)) from the ESP32-S3
to the Waveshare **ESP32-P4-ETH** board. Everything runs on-chip; the model
is ~17 MB of int8 weights.

## Board vs the S3 original

| | S3 demo | this port |
|---|---|---|
| mic | INMP441 on I2S GPIOs | onboard SMD mic via **ES8311** codec (I2C0: SCL=8 SDA=7; I2S0: MCK=13 BCK=12 WS=10 DOUT=9 DIN=11; PA=53 unused) |
| model store | flash partition, mmap'd | see below - one model per resource |
| button | external, GPIO7 | **BOOT = GPIO35** (push-to-talk, active low) |
| LED | WS2812 on RMT | none - `ACT led ...` lines are printed only, so the TUI works unchanged |
| console | USB-Serial-JTAG | **CH343P UART bridge** on /dev/ttyACM1 (115200) |

## Model placement (two resources, two schemes)

| model | file | where | how |
|---|---|---|---|
| Whistle | `whistle.cact` 16.9 MB | SD card `/sdcard/Whistle/whistle.cact` | read into **PSRAM** at boot (~2 MB/s), `EVT model source=sd` |
| Needle 2 | `needle2.cact` 13.7 MB | flash partition `needle2` (data, subtype 0x40, @ 0x410000, 14 MB) | flashed once with esptool, **memory-mapped in place** (the S3 scheme; ready for the needle2 stage) |

```bash
# flash the needle2 model into its partition (one-time, ~4 min at 921600)
python -m esptool --chip esp32p4 -p /dev/ttyACM1 -b 921600 \
    write_flash 0x410000 /path/to/needle2.cact
```

The loader (`model_load()` in `whistle_demo/main/main.c`) tries the SD card
first and falls back to mmap'ing a `whistle` flash partition if the card is
absent - that partition is not allocated by default (32 MB of flash cannot
hold both models), the space is reserved for needle2.

Both apps' `sdkconfig.defaults` enable
`CONFIG_BOOTLOADER_CACHE_32BIT_ADDR_QUAD_FLASH` (+ the experimental-features
flag it depends on): the needle2 partition lives past the 16 MB that 24-bit
flash mapping covers, and without it `esp_partition_mmap` rejects the
address. (Flash an app **with** this setting before the first mmap of
needle2 - a bootloader built without it is fine to skip, the option acts at
app compile time in `flash_mmap.c`.)

## Layout

```
p4/mic_test/        hardware bring-up: ES8311 mic check, levels, PCM dump
p4/whistle_demo/    Whistle STT alone (hold BOOT -> TXT + ACT lines)
p4/needle_demo/     the voice agent: Whistle STT -> Needle 2 LLM -> tool call
p4/components/needle/     shared engine component: globs the engine sources
                          from ../../original/whistle-esp32/engine except
                          nd_quant.c, replaced by the overlay copy in src/
                          (PIE dot128_s8 + boot self-check)
```

The engine itself is unmodified C99 from `original/whistle-esp32/engine`
(keep that clone in place; `original/needle-2-esp32/engine` is not built -
the whistle repo's engine is a superset that also runs needle2.cact, so one
component serves both models and they share the dual-core GEMV worker).
Its only arch-specific path, the `dot128_s8` int8 kernel, has an
ESP32-P4 PIE implementation in the overlay copy (keep the rest of that
file in sync with the original). `p4/needle_demo/main/tools_schema.h.in`
generates the tool schema from `original/needle-2-esp32/tools/led.json` at
build time, exactly like the upstream demo.

## Build & run

ESP-IDF v5.5.x (riscv32 toolchain), Python 3.12 first on PATH:

```bash
export PATH="$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin:$PATH"
source ~/esp/esp-idf/export.sh
cd p4/whistle_demo          # or p4/needle_demo / p4/mic_test
idf.py build && idf.py -p /dev/ttyACM1 flash
```

### whistle_demo

Hold **BOOT**, say a command ("flash the light green"), release. Console
(`EVT`/`TXT`/`ACT` lines, 115200) - keys `2`..`8` change the decoder depth
at runtime, `p` answers `EVT pong` (host liveness probe). The P4 TUI
(`--whistle` mode) follows the live turns:

```bash
.venv/bin/python p4/tools/needle_tui.py --whistle --serial /dev/ttyACM1
```

First live clip on this board: "flash the light green" ->
`TXT Plash the light green.` + `ACT led color=green mode=solid duration=3.0`,
9.2 s for 2.4 s of audio at depth 4 on the portable-C kernel (the S3 with its
Xtensa DSP asm did 6.3 s). `p4/components/needle` now replaces `dot128_s8`
with an ESP32-P4 **PIE** (`xesppie`/`xespv`) kernel: 25 instructions for the
fixed 128-element int8 dot product, double-buffered streams through the
fused `esp.vmulas.s8.xacc.ld.ip` MAC/load/increment form. A boot-time self
check verifies it against the C loop on deterministic vectors and races the
two (`EVT dot128 pie ok ... speedup=25x`); it is bit-identical (integer
accumulation). Measured spoken, depth 4: the C kernel needed 9.2 s for
2.4 s of audio; with PIE the pipeline transcribes **3.6 s of speech in
4.8 s total** (mel 101 ms, stem 561 ms, encoder 2.7 s, cross_kv 91 ms,
decoder 1.4 s, 10 tokens) - ~2.9x end-to-end normalised per second of
audio, and ~2.4x faster than the S3 (which did 6.3 s for 2.0 s).

### needle_demo (the voice agent)

Two models in one firmware: Whistle 16.9 MB from SD -> PSRAM, Needle 2
13.7 MB mmap'd from the flash partition. Boot takes ~1 min (Whistle load
~8 s + one-time prefix priming, 104 tokens / ~40 s with progress lines);
then every request only prefills its own ~15 tokens.

Use it either way:

- hold **BOOT**, say e.g. *"flash red light for two seconds"*, release ->
  `EVT recording/released` + Whistle `EVT clip/speech/lang` + `TXT ...` ->
  Needle 2 streams `TOK ...` (the JSON tool call), `CONF`, and the handler
  prints `ACT led color=red mode=flash duration=2.0`
- or type a request line on the console (same pipeline, no mic);
  `!think` toggles the reasoning block (off by default: verified identical
  calls at half the tokens), a lone digit `2`..`8` still sets the Whistle
  decoder depth

Measured on this board (typed requests, reasoning off), after the LLM
optimizations below: bench **4.87 tok/s (205 ms/tok; the S3 did 1.87** -
2.6x), a full request ~6 s. Five stages got there:

1. **PSRAM staging**: the 2-bit weights stream from flash mmap (~33 MB/s),
   which capped the LUT GEMV. Each tensor is copied to PSRAM on first use
   (hidden in the boot prime; `EVT p4 staged` lines). 2.45 -> 3.0 tok/s.
   The staging table was first capped at 96 entries, which only covered
   layers 0-18 (5.4 MB); the 10.5 MB byte budget was never the limiter.
   Raising the entry cap to cover all 139 lut2 tensors (7.8 MB) staged
   every attention projection + the 2-bit engram tables: 3.83 -> 4.46
   tok/s, proj2bit 127 -> 84 ms, engram 11 -> 4 ms. The same hook now
   also stages the 4-bit mHC phi stacks through `nd_cq_gemv_rows` (3
   tensors, 684 KB; sources already in PSRAM - the SD-loaded Whistle
   model - are skipped by an `esp_ptr_external_ram` guard): 142 entries,
   8.5 MB total, mhc_phi4 32 -> 24 ms, 4.46 -> 4.64 tok/s. (Stage 3
   later dropped the phi staging again: the dec4 cache replaces it, so
   140 staged entries + 1.4 MB of dec4, 9.3 MB of the 10.5 MB budget.)
2. **PIE bit-plane kernel**: PIE has no gather, so the 2-bit LUT path is
   decomposed bilinearly (cb[lo+2hi] expands to 4 coefficients) and the
   three bit-plane sums (lo, 2*hi, lo*hi via `vcmp.eq.s8` masks) come out
   of `esp.vmulas.s8.xacc` MACs over int8-quantised activations, ~120
   instructions per 128 weights. x is quantised per group (the same scheme
   the engine's int8 path uses) and permuted to match uniform byte shifts.
   Guarded by two checks: a synthetic kernel check at boot (exact asm sums
   + <2% composed error) and a first-gemv comparison against the float LUT
   path (<10%; real quant noise is ~3%) - failing either falls back to the
   float path automatically. 3.0 -> 3.7 tok/s, prime 40 -> 25 s.
3. **PIE decoded-rows kernel for the 4-bit phi GEMV**: the mHC phi
   weights are constant across tokens, so each row is decoded once from
   4-bit to int8 (the same per-row decode the engine's int8 GEMM uses)
   and cached in PSRAM (`EVT p4 dec4`); per token the GEMV is then
   `dot128_s8` groups over an int8-quantised xh (side tables rebuilt
   once per layer, keyed on a prepare sequence counter). The remaining
   mhc_phi4 time was ~7.5 ms of engine FWHT in `nd_cq_prepare`
   (addressed by stage 4), ~6 ms of PSRAM-bound PIE dots, ~3 ms
   activation quantise.
   Verified like the lut2 kernel (synthetic boot check + first-gemv
   compare, 1.3% real error); the packed phi blobs are no longer staged
   (dec4 decodes straight from the flash mapping), freeing 684 KB of
   the staging budget. 4.64 -> 4.77 tok/s, mhc_phi4 23.7 -> 16.9 ms.
4. **Unrolled bit-identical FWHT**: every prepare (phi, q_proj, out_proj,
   embedding) runs 16 groups of FWHT(128), and the layer "hadamard"
   sandwich runs two FWHT(512) - all through the engine's `nd_fwht`, one
   butterfly at a time with load->store serialization (~12 cycles per
   butterfly). Running the same butterflies in the same stage order but
   batched (all loads before stores) and unrolled is bit-identical by
   construction - and provably so on device: the rows4 first-gemv verify
   prints the exact same numbers as before the change (worst=157.36133,
   mean=12090.36133). A boot check memcmp-verifies random data against
   the scalar loops and falls back if they ever differ (`EVT p4 fwht ok`,
   which also times both: 13.5 -> 9.5 us per FWHT(128)). 4.77 -> 4.87
   tok/s: mhc_phi4 16.9 -> 15.1 ms, hadamard 12.4 -> 11.3 ms, prep+lut
   9.9 -> 8.9 ms. The transform is load-latency-bound now (~1.45x is all
   scalar C has in it); proj2bit (84 ms, 41%) dwarfs the rest.

The denser staging made two latent Whistle bugs fatal (typed requests were
fine; voice turns died). Both are fixed where they live:

- `nd_quant.c` big-batch tiles (~33 KB) allocated INTERNAL-only with no
  NULL guard: once internal SRAM fragmented, the stem gemms crashed with a
  store to NULL. Tiles are now internal-first with a PSRAM fallback
  (`nd_quant_scratch`), like the activation buffers.
- `nd_whistle.c` decode sessions: the error path leaked every allocation
  that had already succeeded (~1.7 MB per failed turn), and the KV caches
  were sized `dec_layers x max_seq(320)` although a turn decodes at most
  `min(ids cap, 8 + T/2)` positions - so `session_begin` failed for good
  at ~3.3 MB free (`ERR decode`). It now frees on failure and clamps
  `max_seq` to the caller's ids cap for the session. An `EVT decmem` line
  before decode reports free/largest PSRAM block for field checks.

### The TUI

`p4/tools/needle_tui.py` (adapted from the upstream whistle-esp32 TUI -
the superset copy) renders the whole agent live: boot + priming bar,
`ready`, the heard transcript with its language, the streaming tool call,
confidence, timings, and the LED swatch for the resulting `ACT`.

```bash
.venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1            # watch (voice)
.venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1 --reset    # reboot + watch boot
.venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1 \
    --send "make the light yellow for three seconds"                     # typed turn, exits
.venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1 --whistle  # whistle_demo firmware
```

There is no typed REPL: the display runs live because a voice turn can
start at any moment. On connect the TUI probes the board with `p` /
`EVT pong`; if the port-open glitch latched the chip into download mode
(silent), it pulses RTS and follows the clean reboot automatically. The
stats panel always shows the board state - `board` (status · mic ·
bench), `stt` / `llm` model lines with free PSRAM, and the last protocol
line with its age - and after attaching to an already-booted board the
TUI sends `s`, to which both firmwares answer with a recap of their
`EVT ready` / bench / mic summary lines (a plain `status` works on the
needle_demo console too). `ready` in the header is green once both
models are up.

## Gotchas found the hard way

- The USB port is a **CH343P UART bridge**, not USB-Serial-JTAG:
  `CONFIG_ESP_CONSOLE_UART_DEFAULT=y` (already in both sdkconfig.defaults).
- The TF slot is powered by the **on-chip LDO channel 4** - it must be
  attached to the SDMMC host or the card never answers (`send_op_cond
  0x107`). Done in `model_load()`.
- **Flash mmap beyond 16 MB needs 32-bit cache mapping** (see Model
  placement); silently dropped by Kconfig unless
  `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` is also set.
- Long file names: `CONFIG_FATFS_LFN=y` (the card has `Whistle/` and
  `Needle2/` directories).
- After changing `sdkconfig.defaults`, delete the generated `sdkconfig`.
- Only one process may read /dev/ttyACM1; pyserial's open-time DTR/RTS
  glitch resets the board - sometimes latching it *silent* into download
  mode. The TUI probes with `p`/`EVT pong` and recovers by pulsing RTS.

## Status

- [x] mic_test: ES8311 mic verified (left slot, -48 dBFS quiet room)
- [x] whistle_demo: end-to-end transcription from SD->PSRAM model
- [x] needle2 partition + model flashed to flash (mmap loader in place)
- [x] needle_demo: Whistle -> Needle 2 -> grammar-constrained tool call
      (typed verified end-to-end, incl. a second request reusing the
      primed prefix; spoken verified too: "flash red light for two
      seconds" -> TXT -> tool call -> ACT)
- [x] TUI: p4/tools/needle_tui.py - live view of boot/priming/voice turns
      incl. the tool-call panel and LED swatch; typed turns via --send;
      auto-recovery from the port-open latch
- [x] PIE `dot128_s8` on P4 (`p4/components/needle/src/nd_quant.c` overlay):
      bit-identical to C, kernel race 25x, boot self-check in both apps;
      spoken-verified: 4.8 s total for 3.6 s of speech (~2.9x end-to-end,
      ~2.4x faster than the S3)
- [x] LLM speed: PSRAM staging (139 lut2 tensors + the 3 4-bit mHC phi
      stacks, 8.5 MB total) + PIE bit-plane kernel for the 2-bit LUT GEMV
      (`nd_cq_gemv_lut2`) + PIE decoded-int8-rows kernel for the 4-bit
      phi GEMV (`nd_cq_gemv_rows`) + the unrolled bit-identical FWHT -
      2.45 -> 4.87 tok/s (~2.6x the S3),
      prime 40 -> ~21 s; verified with typed requests incl. a CONF 0.96
      call; automatic fallback to the float path if any self-check fails
- [ ] optional: depth/gain experiments; more LLM speed (next target:
      PSRAM bandwidth - proj2bit ~84 ms of the 205 ms token)
