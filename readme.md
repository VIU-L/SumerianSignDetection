**ABOUT**  
Computer Vision coursework project in Ecole polytechnique.  
Author: MENG Yanxu, ZOU Yuran.  
Status: Not finished.  

**GOAL**  
Detect bounding boxes for Sumerian cuneiform signs.

**DATA**  
1. First you need to download the HeiCuBeDa image dataset:
https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/DATA/IE8CCN/X6APKT&version=2.0  
and put the contents into HeiCuBeDa folder.  
You should make sure that the path becomes:  
```
(root of project)   
└── HeiCuBeDa   
    ├── Images_MSII_Filter   
    │   └── ...   
    └── HeiCuBeDa_B_Hilprecht_Database_240121.json   
```
2. The annotations in MaiCuBeDa are based on MaiCuBeDa:   
https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/DATA/QSNIQ2/OXQBR4&version=1.1  
*(No download required)*.  
```all_photo_anno.json, train_photo_anno.json, test_photo_json```: bbox annotation of each photo. Each bbox has a "charname" and a "transliteration".  
```charname_to_id.json, transliteration_to_id.json```: established dict mapping textual categories to int.

**DEVELOPPER NOTES**  
1. Specific goals? (Only Detect bboxes of signs? Detect+Classify them? Should we use charname or transliteration?)
2. Framework? (one-step Faster-RCNN style? First-detect-then-classify?)
3. More advanced objectives? (Based on detections, generate whole transcription of text? If we want transliteration, how do we infer correct reading of a sign? Attention module? May require more NLP.)

**INSPIRATIONS**  
Valuable papers:   
1. https://ieeexplore.ieee.org/document/10350498  
The dataset contributor. They also made a RCNN style framework, but did not publish code.
2. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039
An old paper. Detected lines first; used sign size as first heuristics; then corrects guesses by looking at transliteration. Used graph-based optimizations to assign signs to right place in line and line to right place on photo (minimize loss between sign dispersion and GT transliteration). Did publish code, but with python2.7 and C++ dependencies.
3. https://www.researchgate.net/publication/373643935_R-CNN_based_Polygonal_Wedge_Detection_Learned_from_Annotated_3D_Renderings_and_Mapped_Photographs_of_Open_Data_Cuneiform_Tablets
Same team as (1). An earlier work with similar method (also performed wedge detection). The dataset had been smaller at that moment. Did not publish code.
4. https://cs231n.stanford.edu/2024/papers/img2sumglyphs-transformer-based-ocr-of-sumerian-cuneiform.pdf
Stanford CV coursework project. Brute-force TrOCR on entire tablets. Used lineart and not photos. Did not publish code.