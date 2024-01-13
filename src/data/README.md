# Data Config 

Currently supported options: 


```
additional_feature_engineering: 
    - P_TOT/P_LH
    - P_TOT 
    - aspect_ratio 
additional_mapping_functions: None
feature_filtering: 
    - non-disruptive
    - only-hmode
    - no-impurities
    - no-nbi
    - no-ecrh
    - no-icrh
```

# Anomaly detections

1. Based on the filter-by-time, we take discharge times where the IDA data exists, 
    - Therefore, if machine parameters are interpolated across that domain, we should check how the interpolation occurs


