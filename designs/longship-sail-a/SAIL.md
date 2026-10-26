# Sail A: the glider

An undyed wool sail with a madder-red edging, divided by red cords into a 3x3 grid. Five cells carry a red roundel:

```
. o .
. . o
o o o
```

That is the glider from Conway's Game of Life, laid out exactly like the hacker emblem (Eric S. Raymond's glider in a tic-tac-toe grid). To anyone else it reads as a sail with shield-like roundels, which echo the shields on the rail.

Only the sail differs from `designs/longship/`; the scene, ship, motion and palette are the same. Rebuild with `python3 generate.py` (the sail is chosen by `SAIL = 'a'` near the top).
