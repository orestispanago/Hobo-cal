# Hobo sensors intercomparison

## Outlier removal methods: 

Modified Thompson test

Boxplot rule (IQR)


## Functions in calibration.py:


### Calculations:
    
- Remove outliers

- Mark outliers as True/False in separate dataframe column (use with lmplot)

- Linear regression parameters with standard errors (can be saved to excel)

### Plotting:

- **scatter mattrix** - lower triangle, no diagonal - annotated with Pearson's R or linear regression equation (1 image)
- **scatter plots** - all sensors with reference - annotated with linear regression equation (1 image)
- **diurnal variation plot** -  (1 image)
- **linear model plots** - outliers and non-outliers on same plot (multiple images)
- **categorical plot** - each regression parameter for both methods (multiple images)

---
Other repository contents:

 - /raw/*.csv      : sensors data.
 - plot2page.tex   : [Overleaf](https://www.overleaf.com/) template to arrange multiple plots on one page
