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

## Machine parameters 


## Profiles 

Checking for rapid changes in profiles via: if there are more than 30 slices which have an change in profile average of density/temperature that is > 3stds  of the average change of the averaged profile over whole pulse and the change in profile average is > 0.5 (normalized to keV and 1e-19) discard the pulse. 

# Interpolation 

Based on the filter-by-time, we take discharge times where the IDA data exists. When machine parameters are not given for times below/above where IDA exists, we just take the nearest point: 
`f = interp1d(mp_raw_time, mp_raw_data, bounds_error=False, fill_value=(mp_raw_data[0], mp_raw_data[1]))`