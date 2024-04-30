import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import statsmodels.api as sm

def calculate_vif(X, y):
    """
    Calculates Variance Inflation Factors (VIF) for each feature in a dataset.

    This function adds a constant term to the predictor matrix, computes VIF for each feature, and returns a DataFrame listing each feature alongside its VIF.

    Parameters:
    - X (DataFrame): DataFrame containing predictor variables.
    - y (Series): Series containing the target variable. Note: `y` is not used in the function and can be removed.

    Returns:
    - DataFrame: A DataFrame with two columns: 'VIF', containing the VIF values, and 'col', listing the corresponding feature names.
    """
    
    # Add a constant term (intercept) to the feature matrix
    X_with_const = sm.add_constant(X)
    
    vif_vals = []
    for col in X_with_const.columns:
        vif = variance_inflation_factor(
            X_with_const.values,
            X_with_const.columns.get_loc(col)
        )
        vif_vals.append(vif)

    # Create a DataFrame with variable names and VIF values
    vif_df = pd.DataFrame({'VIF': vif_vals, 'col' : X_with_const.columns})
    
    return vif_df

def remove_high_vif_features(X, y, vif_threshold=5.0):

    # Keep track of columns to remove
    columns_removed = []

    # Logic to calculate VIF and remove high VIF features
    while True:
        # Calculate VIF here
        vif_df = pd.DataFrame({
            'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            'column': X.columns
        })

        # Filter VIFs greater than the threshold
        high_vif = vif_df[vif_df['VIF'] > vif_threshold]
        if high_vif.empty:
            break

        # Remove the column with the highest VIF
        max_vif_column = high_vif.sort_values('VIF', ascending=False).iloc[0]['column']
        X = X.drop(columns=[max_vif_column])
        columns_removed.append(max_vif_column)

    return X, columns_removed