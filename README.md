# ESDTIs
This repo covers research regarding Extremely-Short-Duration Transient Interactions (ESDTIs)

## Overview
ESDTIs are defined in this upcoming paper as interactions between a transiting body, often a satellite, and a star. All observations in this study are done via publicly available Catalina Sky Survey (CSS) information. The preliminary investigation is done with 994 total observations:

| Category      | Number |
| ------------- | ------ |
| Baselines     | 981    |
| Transit Frame | 13     |
| Total         | 994    |

This preliminary group is still being researched, and in-progress files are available for some of the information on ESDTI impact research so far.

## FILESYSTEM!!
The filesystem for the files is organized in a few different folders, so here is the full filesystem layout, as is also described in the full-text:
```
└── Main Parent Folder  
   └── FITS  
      ├── Baseline (Images from CSS with no ESDTIs)  
      ├── Training (Images intended to be used for training)  
      ├── Transit (Images from CSS WITH ESDTIs)  
      ├── Unlabeled (Images that are not yet labeled)  
   └── Figs (Backup for project figures)  
   └── venv  
      ├── Codebase Environment (python venv)  
```

### Notes:
- To-do list and roadmap can be found in the project connected to this repository!
