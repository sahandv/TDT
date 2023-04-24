# TDT Data

TDT can work with different data sources as long as they conform to the format required in each module.

This directory contains some of the API samples, in addition to pre-processing scripts for data cleanup, for the Scopus data.

Please note that to avoid any licensing issue with data, the data or the models may not be shared in this repository at the time of writing. However, they might be shared later as the issues are resolved.

## Ontology Data

The ontology data for computer science is accessible throught this link: https://cso.kmi.open.ac.uk/downloads

You may download  it in an appropriate format (i.e. csv) for this work.


## Scopus data

As discussed earlier, the Scopus data will not be shared directly. The data has been fetched since 1960 to 2021, for `artificial intelligence` and `AI` keyword searchs. 

### Scopus citations data

The citation data has been also downloaded using the API for scopus. This task is comparatively long, as there is a limitation on the references we can download at a given time. The references should be downloaded in a loop of papers, for each paper. Then, they should be unified into a single citations data. This data will be shared in this repository soon.

### Scopus abstrac and keyword data

The data for abstract and keywords of scopus will not be shared. However, the trained FastText and Doc2Vec models will be shared after resolving the issues with licensing.


