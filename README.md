# Segmentation Analyzer

Tools for analyzing and visualizing properties of generated segmentations from the RPE-map project. Various shape
metrics for cell, nucleus and organelle are calculated. For example: Volume, minimum and maximum feret, spans in x,y and
z direction, Maximum Intensity Projection area, etc.

These values are stored in .npz files that can be later retrieved for plotting.
Also included is code for generating a single synthetic cell with an organell
e for validation of results.

Metadata files are also generated to store the data. These may be used for redoing calculations or locating certain
cells by ID for visualization or other purposes.

## System requirements and Installation.

Requirements depend upon the size of data being used. It is recommended (but not necessary) that you have at least 16 Gb
ram and a few hundred Gb of free space to accommodate the data.

1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   or [Anaconda](https://www.anaconda.com/products/individual).


2. Clone **SegmentationAnalyzer** to a local directory `<path>/SegmentationAnalyzer`.
   (Where `<path>` is the local directory of your choosing). You can do this by navigating to the location and using the
   following command in your command prompt or terminal:

   ```
   git clone https://github.com/NIH-NEI/SegmentationAnalyzer
   ```

<ul>
 Alternatively, you may simply download zip under the code button on the same webpage.
</ul>

3. Run Anaconda Prompt/Terminal or in your IDE, cd to `<path>/SegmentationAnalyzer`.


4. Create Conda Virtual Environment (do this once on the first run):

   ```
   conda env create --file conda-environment.yml
   ```

5. Activate the Virtual Environment:

   ```
   conda activate SegmentationAnalyzer
   ```

## Data and Demo

For a quick demonstration of a single calculation, you can run

```
python analysis/AnalysisTools/SyntheticData.py
```

This will generate a single synthetic polygonal prism cell, and an elliptical organelle in a separate channel. Then
calculate metrics for the data. Note that for organelles, a maximum limit of 250 organelles is set and for most
properties, a matrix of that length is generated. In case there is only one organelle as in this case, you can ignore
the other 249 nan columns.

Data used for the experiment can be found at [Deepzoomweb: RPEmap](https://isg.nist.gov/deepzoomweb/data/RPEmap). For
this repository, we use the segmented data that is hosted in the above link. Create a main folder, and create a
subfolder for each organelle. GFP segmentations are kept in <path-to-gfp folder> and their corresponding Actin and DNA
segmentations should be in <path-to-corresponding-segmented-cell-folder>. 

## Running the code

### Calculations

To run calculations for segmented stacks, run GenerateShapeMetricsBatch.py as follows:

```
(<path-to-environment>/)python <path-to_project>/GenerateShapeMetricsBatch.py --GFPFolder <path-to-gfp folder> --CellFolder <path-to-corresponding-segmented-cell-folder> --savepath <savepath> --channel <channelname>
```

This code outputs '.npz' files with filenames `<gfpchannel>_<organelle>_<metric>.npz` for each metric, where gfpchannel
refers to the organelle tagged by gfp channel, and organelle refers to the gfp channel, DNA or Actin. Metrics calculated
include:
> Cell Metrics:
>> Centroid, Volume, X span, Y span, Z span, MIP area, Max feret, Min feret,
> > Mean feret, 2D Aspect ratio, Sphericity

> DNA metrics:
>> Centroid, Volume, X span, Y span, Z span, MIP area, Max feret, Min feret,
> > Mean feret, 2D Aspect ratio, Volume fraction, Sphericity, z-distribution

> GFP Metrics:
>> Centroid, Volume, X span, Y span, Z span, MIP area, Max feret, Min feret,
> > Mean feret, 2D Aspect ratio, Volume fraction, Count per cell, etc

To get help information on what the parameters mean, you can run:

```
python GenerateShapeMetricsBatch.py --help
```

### Plotting

Load and plot cell and organelle data over 4 weeks and 2 treatments (set in src.experimentalparams):

```
python loadandplot.py
```

Similar to the calculating metrics, to get information on the parameters, you can run:

```
python loadandplot.py --help
```

## Contributors

* *Pushkar Sathe*
* *Nicholas Schaub* (pseudo-hull based feret calculation)
* *Andrei Volkov* (version compatibility, API access from external Python projects)
