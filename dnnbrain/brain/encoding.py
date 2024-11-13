from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from os.path import join as pjoin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_correlations(model, test_feature, test_brain):
    """
    Compute the correlation between the predicted and actual brain activity.

    Parameters:
    - model: The trained regression model.
    - test_feature: The test set features.
    - test_brain: The test set brain activity.

    Returns:
    - correlations: Correlation between predicted and actual brain activity.
    """
    predictions = model.predict(test_feature)
    std_predictions = (predictions - predictions.mean(axis=0)) / predictions.std(axis=0)
    std_brain_test = (test_brain - test_brain.mean(axis=0)) / test_brain.std(axis=0)

    correlations = np.sum(std_predictions * std_brain_test, axis=0) / (std_predictions.shape[0] - 1)
    return correlations


def encoding_model(train_brain=None, test_brain=None, train_feature=None, test_feature=None, 
                   brain_data=None, voxel_mask=None, feature=None, cv=True, n_splits=10, n_jobs=8, 
                   feature_type=None, add_flag=None, save_map=True, save_weights=True, save_results=True):
    """
    Train an encoding model using cross-validation or specified train/test data.

    Parameters:
    - train_brain: Training brain data (if not using cross-validation)
    - test_brain: Testing brain data (if not using cross-validation)
    - train_feature: Training feature data (if not using cross-validation)
    - test_feature: Testing feature data (if not using cross-validation)
    - brain_data: The full brain data (used only when cross-validation is enabled)
    - voxel_mask: Bool array to constrain the target ROI
    - feature: The full feature data (used only when cross-validation is enabled)
    - cv: Boolean, whether to perform cross-validation (default: True)
    - n_splits: Number of folds for cross-validation (default: 10)
    - n_jobs: Number of jobs for parallel processing (default: 8)
    - feature_type: Name of the feature_type: {dataset_name}_{model_name}_{layer_name}_{method}
    - add_flag: Name tp specify the voxel mask
    - save_map: Boolean, whether to save the encoding map (default: True)
    - save_weights: Boolean, whether to save the model weights (default: True)
    - save_results: Boolean, whether to save the results (default: True)
    """
    # preprocess data
    if voxel_mask is not None:
        brain_data = brain_data[:, voxel_mask]
    brain_data = zscore(brain_data, axis=1)

    # process on add flag
    if add_flag is not None:
        add_flag = '_' + add_flag
    else:
        add_flag = ''

    # If performing cross-validation, use KFold
    kf = KFold(n_splits=n_splits) if cv else None

    correlations_list = []
    weights_sum = None  # Initialize the variable to store the sum of weights for averaging later

    # Depending on whether CV is used, either iterate over CV splits or directly fit the model
    if cv:
        for train_idx, test_idx in tqdm(kf.split(brain_data), desc='Performing cross-validation'):
            # Split the brain and feature data for training and testing
            train_brain_cv, test_brain_cv = brain_data[train_idx], brain_data[test_idx]
            train_feature_cv, test_feature_cv = feature[train_idx], feature[test_idx]

            # Fit the model
            lr = make_pipeline(StandardScaler(), LinearRegression(n_jobs=n_jobs))
            lr.fit(train_feature_cv, train_brain_cv)

            # Compute correlations
            correlations = compute_correlations(lr, test_feature_cv, test_brain_cv)
            correlations_list.append(correlations)
            print(f'{feature_type}, r mean={correlations.mean(axis=0):.3f} r max={correlations.max():.3f} in this cv')

            # Accumulate weights for averaging later
            linear_regression_step = lr.named_steps['linearregression']
            if weights_sum is None:
                weights_sum = linear_regression_step.coef_.T  # Initialize with the first set of weights
            else:
                weights_sum += linear_regression_step.coef_.T  # Add subsequent weights
    else:
        # No cross-validation, use provided train/test data
        lr = make_pipeline(StandardScaler(), LinearRegression(n_jobs=n_jobs))
        lr.fit(train_feature, train_brain)

        # Compute correlations
        correlations = compute_correlations(lr, test_feature, test_brain)
        correlations_list.append(correlations)

        # Get weights from the single model
        linear_regression_step = lr.named_steps['linearregression']
        weights_sum = linear_regression_step.coef_.T

    # Compute average correlation across all folds (if CV) or single result (if no CV)
    avg_correlations = np.mean(correlations_list, axis=0)
    lr_r = avg_correlations.mean()
    print(f'{feature_type}, features={feature.shape[1]}, r={lr_r:.3f}')

    # Save encoding model map
    if save_map:
        encoding_map = np.zeros((91282))
        if voxel_mask is None:
            encoding_map[:59412] = avg_correlations 
        else:
            encoding_map[:59412][voxel_mask] = avg_correlations  # Apply the voxel mask
        encoding_path = pjoin(encoding_result_path, f'{feature_type}_encoding_map{add_flag}.dtseries.nii')
        save_ciftifile(encoding_map, encoding_path)

    # Save model weights
    if save_weights:
        # If cross-validation was used, compute the average weights
        if cv:
            avg_weights = weights_sum / n_splits
        else:
            avg_weights = weights_sum

        # Save the average weights
        np.save(pjoin(weight_path, f'{feature_type}_encoding_weights{add_flag}.npy'), avg_weights)

    # Save results
    results_df = pd.DataFrame(index=range(len(avg_correlations)))
    results_df['correlations'] = avg_correlations

    # Save final results to CSV
    if save_results:
        results_df.to_csv(pjoin(encoding_result_path, f'{feature_type}_encoding_correlations{add_flag}.csv'), index=False)
