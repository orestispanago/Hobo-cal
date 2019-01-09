A comparison between sensors is performed using "calibration.py".

Outlier removal methods: Modified Thompson test and Boxplot rule (IQR)

Functions included:

    Calculations:
        
        Outliers are dropped or marked as True/False for both methods
        Linear regression parameters with standard errors (can be saved to excel)

    Plotting:
    
        scatter mattrix - lower triangle, no diagonal - annotated with Pearson's R or linear regression equation (1 image)
        scatter plots - all sensors with reference - annotated with linear regression equation (1 image)
        diurnal variation plot -  (1 image)
        linear model plots - outliers and non-outliers (multiple images)
        categorical plot - each regression parameter for both methods (multiple images)


Other repository contents:

    /raw/*.csv      : sensors data.
    plot2page.tex   : overleaf template to arrange multiple plots to one page
