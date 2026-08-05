# Map2Decibel

TLDR: See map2decibel_demo.ipynb for interactive notebook after cloning. Use the model from the release page. 

# About

This is our attempt to do street-level road noise prediction from OpenStreetMap morphology. It predicts L_den (day-evening-night noise level, dBA) for any city using only freely available OpenStreetMap geometry WITHOUT any traffic counts, no sensors, no specialist acoustic software.


## Local run
 
```bash
pip install -r requirements.txt
```
 
## Limitations
 
- Not meant to be replacement of real sensors. 
- OSM data quality varies across regions.
*Paper under blind review.*

