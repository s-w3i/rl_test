# rl_test
reinforcement learning test

## Configurable obstacle map

`multi_astar.py` and `single_path.py` read their obstacle grids from a plain
text file.  This repository includes a default `map.txt` that both scripts load
automatically.  To use a different map, set the `MAP_FILE` environment variable
to another file path.  Each map file should contain rows of `0` and `1`
separated by spaces, where each line corresponds to one row in the grid.  For
example:

```
1 1 1 1
1 0 0 1
1 0 0 1
1 1 1 1
```

If the specified file cannot be read, the scripts raise a clear error message.

`single_path.py` randomly chooses six start/goal pairs from the map and runs A*
between each pair.  When executed it opens a Matplotlib window and animates all
six agents moving along their computed paths, each agent shown in a different
color with a numbered label inside a larger marker.
