# Sail B: the bits

At a glance this is the usual striped sail. The cloth is sewn from three horizontal bands, and each band is eight stripe cells wide. Each band is one byte, read left to right with the most significant bit at the left; a stripe-coloured cell (cream in the light theme, red in the dark one) is a 1.

```
band 1   0 1 1 0 1 0 0 0   0x68   h
band 2   0 1 1 0 0 1 0 1   0x65   e
band 3   0 1 1 0 1 0 0 1   0x69   i
```

The sail says "hei", Norwegian for "hello": the ship's own hello world. A side note: the original sail's seven stripe slots, alternating off and on, already happened to read 0101010 = 42 = `*`.

To spell something else, change `WORD` in `generate.py` (keep it short; three or four letters still read well at banner size) and run `python3 generate.py`. Only the sail differs from `designs/longship/`.
