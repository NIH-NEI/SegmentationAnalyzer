## Segmentation Analyzer

Tools for  analyzing and visualizing properties of generated segmentations from the RPE-map project. Various shape metrics for cell, nucleus and organelle are calculated. For example: Volume, minimum and maximum feret, spans in x,y and z direction, Maximum Intensity Projection area, etc.

These values are stored in .npz files that can be later retrieved for plotting.

Metadata files are also generated to store the data. These may be used for redoing calculations or locating certain cells by ID for visualization or other purposes. 


## Instructions

1. Install:
```
git clone 
```

2. Load and plot cell and organelle data over 4 weeks and 2 treatments (set in src.experimentalparams): 
```

import loadandplot
```

3. aa:
```

```