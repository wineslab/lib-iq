import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
import os
from collections import defaultdict
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator
from libiq.iq_clustering.preprocessing import load_csv
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.patches as mpatches
from sklearn import metrics as metr
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, List, Union, Tuple
from libiq.utils.constants import PLOTS_PATH, PLOTS_MODE, LABELS, CLUSTER_MODEL_PATH, PLOT_TRAFFIC_FIGURE_SIZE, STATIC_LABELS, PLOT_TRAFFIC_CMAP, CLUSTERS, METRICS, GRIDSEARCH_INIT, N_CLUSTERS
from multiprocessing import Pool, cpu_count

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress specific FutureWarnings related to sklearn
warnings.filterwarnings("ignore", category=FutureWarning, message=".*define the `__sklearn_tags__` method.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

def metrics(x_df_reshaped: np.ndarray, y_pred_reshaped: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> None:
    """
    Calculate and print various clustering metrics to evaluate clustering quality.

    Args:
        x_df_reshaped (np.ndarray): Reshaped data array.
        y_pred_reshaped (np.ndarray): Reshaped prediction array.
        y_pred (np.ndarray): Prediction array for original clustering labels.
        labels (np.ndarray): True labels array for ground truth.

    Raises:
        ValueError: If there is an issue with calculating any of the metrics.
    """

    try:
        unique_labels = np.unique(y_pred)
        
        if len(unique_labels) > 1 and len(y_pred) > 2:
            train_silhouette = metr.silhouette_score(x_df_reshaped, y_pred_reshaped)
            print("\tSilhouette score: {:.2f}\t(Range: -1 to 1, better is closer to 1)".format(train_silhouette))
        else:
            print("\tSilhouette score: Not enough samples to calculate silhouette score")

        rand_score_val = metr.rand_score(labels, y_pred)
        print("\tRand score: {:.2f}\t(Range: 0 to 1, better is closer to 1)".format(rand_score_val))
        
        adjusted_rand_score_val = metr.adjusted_rand_score(labels, y_pred)
        print("\tAdjusted rand score: {:.2f}\t(Range: -1 to 1, better is closer to 1)".format(adjusted_rand_score_val))
        
        mutual_info_score_val = metr.mutual_info_score(labels, y_pred)
        print("\tMutual info score: {:.2f}\t(Range: 0 to ∞, better is higher)".format(mutual_info_score_val))
        
        adjusted_mutual_info_score_val = metr.adjusted_mutual_info_score(labels, y_pred)
        print("\tAdjusted mutual info score: {:.2f}\t(Range: -1 to 1, better is closer to 1)".format(adjusted_mutual_info_score_val))
        
        normalized_mutual_info_score_val = metr.normalized_mutual_info_score(labels, y_pred)
        print("\tNormalized mutual info score: {:.2f}\t(Range: 0 to 1, better is closer to 1)".format(normalized_mutual_info_score_val))

        homogeneity_score_val = metr.homogeneity_score(labels, y_pred)
        print("\tHomogeneity score: {:.2f}\t(Range: 0 to 1, better is closer to 1)".format(homogeneity_score_val))
        
        completeness_score_val = metr.completeness_score(labels, y_pred)
        print("\tCompleteness score: {:.2f}\t(Range: 0 to 1, better is closer to 1)".format(completeness_score_val))

        v_measure_score_val = metr.v_measure_score(labels, y_pred)
        print("\tV measure score: {:.2f}\t(Range: 0 to 1, better is closer to 1)".format(v_measure_score_val))

        fowlkes_mallows_score_val = metr.fowlkes_mallows_score(labels, y_pred)
        print("\tFowlkes mallows score: {:.2f}\t(Range: 0 to 1, better is closer to 1)".format(fowlkes_mallows_score_val))

        if len(unique_labels) > 1 and len(y_pred) > 2:
            calinski_harabasz_score_val = metr.calinski_harabasz_score(x_df_reshaped, y_pred_reshaped)
            print("\tCalinski Harabasz score: {:.2f}\t(Higher is better)".format(calinski_harabasz_score_val))
        else:
            print("\tCalinski Harabasz score: Not enough samples to calculate Calinski Harabasz score")

        if len(unique_labels) > 1 and len(y_pred) > 2:
            davies_bouldin_score_val = metr.davies_bouldin_score(x_df_reshaped, y_pred_reshaped)
            print("\tDavies Bouldin score: {:.2f}\t(Range: 0 to ∞, better is closer to 0)".format(davies_bouldin_score_val))
        else:
            print("\tDavies Bouldin score: Not enough samples to calculate Davies Bouldin score")
    except Exception as e:
        raise ValueError(f"An error occurred while calculating metrics: {e}")

def print_scatterplot_3d(x: np.ndarray, labels: List[int], x_magnitude: np.ndarray = None, mode: str = 'scatter', path: str = '') -> None:
    """
    Plot a 3D scatter plot of the real and imaginary parts of the data, optionally including magnitude.

    Args:
        x (np.ndarray): Input array containing real and imaginary parts of the data (complex representation).
        labels (List[int]): List of cluster labels for each sample.
        x_magnitude (np.ndarray, optional): Magnitude values corresponding to `x` data.
        mode (str): Plotting mode; 'scatter' for scatter plot or 'plot' for continuous line plot. Defaults to 'scatter'.
        path (str): Path to save the plot as an image file. Defaults to '' (no saving).

    Raises:
        ValueError: If an invalid plotting mode is specified or if plotting fails.
    """

    try:
        if x_magnitude is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            unique_labels = np.unique(labels)
            custom_colors = ['gray', 'violet', 'lightgreen', 'orange']
            if len(unique_labels) > len(custom_colors):
                raise ValueError("Not enough colors for the unique labels. Add more colors to custom_colors list.")
            label_color_map = {label: color for label, color in zip(unique_labels, custom_colors)}

            for label in unique_labels:
                indices = [i for i, l in enumerate(labels) if l == label]
                for index in indices:
                    num_samples = np.arange(x.shape[1])
                    real_data = x[index, :, 0]
                    imaginary_data = x[index, :, 1]
                    color = label_color_map[label]

                    cluster_label = f'cluster{label}'
                    if mode == 'scatter':
                        ax.scatter(num_samples, real_data, imaginary_data, label=cluster_label if index == indices[0] else "", color=color)
                    elif mode == 'plot':
                        ax.plot(num_samples, real_data, imaginary_data, label=cluster_label if index == indices[0] else "", color=color)
                    else:
                        raise ValueError(f"Invalid mode: {mode}. Choose 'scatter' or 'plot'.")

            ax.set_xlabel('Number of Samples')
            ax.set_ylabel('Real Part')
            ax.set_zlabel('Imaginary Part')
            ax.set_title('3D Plot of Real and Imaginary Components')
            ax.legend()
            plt.show()
        else:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            unique_labels = np.unique(labels)
            custom_colors = ['gray', 'violet', 'lightgreen', 'orange']
            if len(unique_labels) > len(custom_colors):
                raise ValueError("Not enough colors for the unique labels. Add more colors to custom_colors list.")
            label_color_map = {label: color for label, color in zip(unique_labels, custom_colors)}

            for label in unique_labels:
                indices = [i for i, l in enumerate(labels) if l == label]
                for index in indices:
                    real_data = x[index, :, 0]
                    imaginary_data = x[index, :, 1]
                    magnitude_data = x_magnitude[index]
                    color = label_color_map[label]

                    cluster_label = f'cluster{label}'
                    if mode == 'scatter':
                        ax.scatter(magnitude_data, real_data, imaginary_data, label=cluster_label if index == indices[0] else "", color=color)
                    elif mode == 'plot':
                        ax.plot(magnitude_data, real_data, imaginary_data, label=cluster_label if index == indices[0] else "", color=color)
                    else:
                        raise ValueError(f"Invalid mode: {mode}. Choose 'scatter' or 'plot'.")

            ax.set_xlabel('Magnitude (Db)')
            ax.set_ylabel('Real Part')
            ax.set_zlabel('Imaginary Part')
            ax.set_title('3D Plot of Real and Imaginary Components')
            ax.legend()

            plt.show()

    except Exception as e:
        raise ValueError(f"An error occurred while plotting scatter plot: {e}")

def plot_silhouette(model: Any, x_df: pd.DataFrame, path: str = '') -> None:
    """
    Plot silhouette analysis for clustering results.

    Args:
        model (Any): Clustering model to use for silhouette analysis.
        x_df (pd.DataFrame): Input data for clustering.
        path (str): Path to save the plot. If empty, the plot is displayed. Defaults to ''.

    Raises:
        ValueError: If an issue occurs during silhouette analysis or plotting.
    """

    try:
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                visualizer.fit(x_df)
        if PLOTS_MODE == 'interactive':
            visualizer.show()
        else:
            if path != '':
                visualizer.show(outpath=path, format='pdf')
                plt.close()
    except Exception as e:
        raise ValueError(f"An error occurred while plotting silhouette: {e}")

def plot_traffic_type_distribution(y_df: Union[pd.DataFrame, pd.Series, dict, list], y_pred: Union[pd.DataFrame, pd.Series, dict, list, np.ndarray], n_clusters: int, path: str = '') -> None:
    """
    Plot the distribution of true and predicted labels across different traffic types.

    Args:
        y_df (Union[pd.DataFrame, pd.Series, dict, list]): True labels.
        y_pred (Union[pd.DataFrame, pd.Series, dict, list, np.ndarray]): Predicted cluster labels.
        n_clusters (int): Number of clusters in the prediction.
        path (str): Path to save the plot image. Defaults to ''.

    Raises:
        ValueError: If there is a mismatch in length between true and predicted labels or if plotting fails.
    """

    try:
        if isinstance(y_df, list):
            y_df = pd.DataFrame(y_df, columns=['labels'])
        elif isinstance(y_df, pd.Series):
            y_df = y_df.to_frame(name='labels')
        elif isinstance(y_df, dict):
            y_df = pd.DataFrame(list(y_df.values()), columns=['labels'])
        
        if isinstance(y_pred, list):
            y_pred = pd.DataFrame(y_pred, columns=['pred'])
        elif isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_frame(name='pred')
        elif isinstance(y_pred, dict):
            y_pred = pd.DataFrame(list(y_pred.values()), columns=['pred'])
        elif isinstance(y_pred, np.ndarray):
            y_pred = pd.DataFrame(y_pred, columns=['pred'])

        assert len(y_df) == len(y_pred), "Mismatch in number of rows between y_df and y_pred"

        labeled_df = pd.DataFrame({
            'labels': y_df['labels'].values,
            'pred': y_pred['pred'].values
        })

        ##########################
        #CHE SENSO HA FARE QUESTO MAPPING SE PI NON SO CLUSTER 0 A QUALE TECNOLOGIA CORRISPONDE? CHI MI DICE CHE CLUSTER = SIAESATTAMENTE 5G E NON WIFI??
        #INTENDO LE TRE RIGHE SUBITO SOTTO
        #E lA RIGA DU LABEL NAME E XTICK DEL GRAFICO IN FONDO
        #ATTUALMENTE HO DIsATTIVATO IL MAPPING CON LE TECNOLOGIE
        ##########################
        label_map = {label: i for i, label in enumerate(np.sort(labeled_df['labels'].unique()))}
        labeled_df['labels'] = labeled_df['labels'].map(label_map)
        labeled_df['pred'] = labeled_df['pred'].map(lambda x: x % n_clusters)

        fig, ax = plt.subplots(figsize=PLOT_TRAFFIC_FIGURE_SIZE)

        unique_labels = np.sort(labeled_df['labels'].unique())
        unique_preds = np.arange(n_clusters)
        cmap = plt.cm.get_cmap(PLOT_TRAFFIC_CMAP, n_clusters)
        pred_colors = {i: cmap(i) for i in range(n_clusters)}
        
        #label_names = {i: STATIC_LABELS.get(i, f'Label {i}') for i in range(len(unique_labels))}

        contingency_table = pd.crosstab(labeled_df['labels'], labeled_df['pred'])
        
        bottom = np.zeros(len(unique_labels))
        for pred in unique_preds:
            counts = contingency_table[pred] if pred in contingency_table else np.zeros(len(unique_labels))
            ax.bar(unique_labels, counts, bottom=bottom, color=pred_colors[pred], label=f'Cluster {pred}')
            bottom += counts

        handles = [mpatches.Patch(color=color, label=f'Cluster {pred}') for pred, color in pred_colors.items()]
        
        ax.legend(handles=handles)
        ax.set_xticks(unique_labels)
        #ax.set_xticklabels([label_names[label] for label in unique_labels], rotation=45, ha='right')
        ax.set_xlabel('Original Labels')
        ax.set_ylabel('# occurrences')
        if PLOTS_MODE == 'interactive':
            plt.show()
        else:
            if path != '':
                plt.savefig(path, format='pdf')
                plt.close()

    except Exception as e:
        raise ValueError(f"An error occurred while plotting traffic type distribution: {e}")

def custom_silhouette_scorer(estimator: Any, x_df: pd.DataFrame) -> float:
    """
    Calculate a custom silhouette score for clustering, suitable for GridSearchCV.

    Args:
        estimator (Any): Clustering estimator (model) for prediction.
        x_df (pd.DataFrame): Input data for clustering.

    Returns:
        float: The calculated silhouette score.

    Raises:
        ValueError: If an issue occurs while calculating the silhouette score.
    """

    try:
        y_pred = estimator.fit_predict(x_df)
        
        if x_df.ndim == 3:
            x_df_reshaped = x_df.reshape(x_df.shape[0] * x_df.shape[1], x_df.shape[2])
            y_pred_reshaped = np.repeat(y_pred, x_df.shape[1])
        else:
            x_df_reshaped = x_df
            y_pred_reshaped = y_pred
        
        unique_labels = np.unique(y_pred_reshaped)
        
        if len(unique_labels) > 1:
            silhouette_avg = silhouette_score(x_df_reshaped, y_pred_reshaped)
            return silhouette_avg
        else:
            return -1
    except Exception as e:
        raise ValueError(f"An error occurred while calculating silhouette score: {e}")

def GridSearch(x_df: pd.DataFrame, random_state: int, jobs: int) -> Dict[str, Any]:
    """
    Perform a grid search to find the optimal parameters for the TimeSeriesKMeans clustering model.

    Args:
        x_df (pd.DataFrame): Input data for clustering.
        random_state (int): Random state for reproducibility.
        jobs (int): Number of jobs for parallel processing.

    Returns:
        Dict[str, Any]: Dictionary of the best parameters found by grid search.

    Raises:
        ValueError: If grid search fails.
    """

    try:
        param_grid = {
            'n_clusters': CLUSTERS,
            'metric': METRICS,
            'random_state': [random_state],
            'init': GRIDSEARCH_INIT,
        }

        model = TimeSeriesKMeans()
        scorer = custom_silhouette_scorer

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scorer, n_jobs=jobs, return_train_score=True, verbose=4)
        grid_search.fit(x_df)

        results = pd.DataFrame(grid_search.cv_results_)
        print("Grid Search Results:")
        print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])

        return grid_search.best_params_
    except Exception as e:
        raise ValueError(f"An error occurred during grid search: {e}")

def k_mean_dba(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, random_state: int, jobs: int, n_clusters: int = 4, grid_search: bool = False, plots: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform DBA k-means clustering on training and testing datasets and evaluate clustering results.

    Args:
        x_train (np.ndarray): Training data features.
        x_test (np.ndarray): Testing data features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        random_state (int): Random seed for reproducibility.
        jobs (int): Number of jobs for parallel processing.
        n_clusters (int, optional): Number of clusters. Defaults to 4.
        grid_search (bool, optional): Whether to use grid search for hyperparameter tuning. Defaults to False.
        plots (bool, optional): Whether to generate and save plots. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cluster labels predicted for training and testing datasets.

    Raises:
        ValueError: If an issue occurs during k-means clustering.
    """

    try:
        print("\nDBA k-means")
        print("\nStarting training")
        
        if grid_search:
            best_params = GridSearch(x_train, random_state, jobs)
            print("Best parameters found: ", best_params)
            model = TimeSeriesKMeans(**best_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred_kmeans_train = model.fit_predict(x_train)
            n_clusters = best_params['n_clusters']
        else:
            model = TimeSeriesKMeans(n_clusters=n_clusters, random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred_kmeans_train = model.fit_predict(x_train)
        
        train_cluster_counts = np.bincount(y_pred_kmeans_train, minlength=n_clusters)
        print("Training set cluster counts:", np.bincount(y_train, minlength=n_clusters))
        print("Training set prediction cluster counts:", train_cluster_counts)

        if plots:
            plot_traffic_type_distribution(y_train, y_pred_kmeans_train, n_clusters)

        reshaped_train_dataset = x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2])
        reshaped_y_pred_kmeans_train = np.repeat(y_pred_kmeans_train, x_train.shape[1])

        metrics(reshaped_train_dataset, reshaped_y_pred_kmeans_train, y_pred_kmeans_train, y_train)

        print("\nStarting test")
        y_pred_kmeans_test = model.predict(x_test)
        test_cluster_counts = np.bincount(y_pred_kmeans_test, minlength=n_clusters)
        print("Test set cluster counts:", np.bincount(y_test, minlength=n_clusters))
        print("Test set prediction cluster counts:", test_cluster_counts)

        if plots:
            plot_traffic_type_distribution(y_test, y_pred_kmeans_test, n_clusters)

        reshaped_test_data = x_test.reshape(x_test.shape[0] * x_test.shape[1], x_test.shape[2])
        reshaped_y_pred_kmeans_test = np.repeat(y_pred_kmeans_test, x_test.shape[1])

        metrics(reshaped_test_data, reshaped_y_pred_kmeans_test, y_pred_kmeans_test, y_test)
        return y_pred_kmeans_train, y_pred_kmeans_test
    except Exception as e:
        raise ValueError(f"An error occurred during k-means clustering: {e}")

'''
def k_mean_dba_magnitude(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, random_state: int, jobs: int, n_clusters: int = N_CLUSTERS, grid_search: bool = False, plots: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform DBA k-means clustering on the magnitude data and evaluate the results.

    Args:
        x_train (np.ndarray): Training data.
        x_test (np.ndarray): Test data.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        random_state (int): Random seed for reproducibility.
        jobs (int): Number of parallel jobs for computation.
        n_clusters (int, optional): Number of clusters for k-means. Defaults to 4.
        grid_search (bool, optional): Whether to perform grid search for best parameters. Defaults to False.
        plots (bool, optional): Whether to generate plots for visualization. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Predicted cluster labels for the training and testing datasets.

    Raises:
        ValueError: If an error occurs during k-means clustering with magnitude.
    """
    try:
        print("\nDBA k-means with magnitude")
        print("\nStarting training")

        if grid_search:
            best_params = GridSearch(x_train, random_state, jobs)
            print("Best parameters found:", best_params)
            model = TimeSeriesKMeans(**best_params)
            y_pred_kmeans_train = model.fit_predict(x_train)
            n_clusters = best_params['n_clusters']
        else:
            model = TimeSeriesKMeans(n_clusters=n_clusters, random_state=random_state)
            y_pred_kmeans_train = model.fit_predict(x_train)

        joblib.dump(model, f'{CLUSTER_MODEL_PATH}/clustering_model.joblib')

        train_cluster_counts = np.bincount(y_pred_kmeans_train)
        print("Training set cluster counts:", np.bincount(y_train))
        print("Training set prediction cluster counts:", train_cluster_counts)

        if plots:
            plot_traffic_type_distribution(y_train, y_pred_kmeans_train, n_clusters)
            plot_silhouette(model, x_train)
        
        metrics(x_train, y_pred_kmeans_train, y_pred_kmeans_train, y_train)

        print("\nStarting test")
        y_pred_kmeans_test = model.predict(x_test)

        if plots:
            plot_traffic_type_distribution(y_test, y_pred_kmeans_test, n_clusters)
            plot_silhouette(model, x_test)
        
        metrics(x_test, y_pred_kmeans_test, y_pred_kmeans_test, y_test)
        return y_pred_kmeans_train, y_pred_kmeans_test

    except Exception as e:
        raise ValueError(f"An error occurred during k-means clustering with Magnitude: {e}")
'''

def k_mean_dba_magnitude_train(x_train: np.ndarray, y_train: np.ndarray, model_path: str, random_state: int, jobs: int, n_clusters: int = N_CLUSTERS, grid_search: bool = False, plots: bool = False) -> np.ndarray:
    """
    Train DBA k-means clustering on magnitude data and save the trained model.

    Args:
        x_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training labels.
        model_path (str): Path to save the trained model.
        random_state (int): Random seed for reproducibility.
        jobs (int): Number of parallel jobs.
        n_clusters (int, optional): Number of clusters. Defaults to N_CLUSTERS.
        grid_search (bool, optional): Whether to use grid search for hyperparameter tuning. Defaults to False.
        plots (bool, optional): Whether to generate and save plots. Defaults to False.

    Returns:
        np.ndarray: Cluster labels for the training dataset.

    Raises:
        ValueError: If clustering fails.
        FileNotFoundError: If the specified path for saving the model does not exist.
    """

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The directory {model_path} does not exist.")

        print("\nStarting clustering training")

        y_pred_kmeans_train = []

        if grid_search:
            best_params = GridSearch(x_train, random_state, jobs)
            print("Best parameters found:", best_params)
            model = TimeSeriesKMeans(**best_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred_kmeans_train = model.fit_predict(x_train)
            n_clusters = best_params['n_clusters']
        else:
            model = TimeSeriesKMeans(n_clusters=n_clusters, random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred_kmeans_train = model.fit_predict(x_train)

        joblib.dump(model, f"{model_path}clustering_model.joblib")
        print(f"Model saved at {model_path}")

        train_cluster_counts = np.bincount(y_pred_kmeans_train)
        print("Training set cluster counts:", np.bincount(y_train))
        print("Training set prediction cluster counts:", train_cluster_counts)

        if plots:
            plot_traffic_type_distribution(y_train, y_pred_kmeans_train, n_clusters, f'{PLOTS_PATH}traffic_type_distribution_train.pdf')
            plot_silhouette(model, x_train, f'{PLOTS_PATH}silhouette_train.pdf')
        
        metrics(x_train, y_pred_kmeans_train, y_pred_kmeans_train, y_train)
        return y_pred_kmeans_train

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        raise
    except ValueError as val_error:
        print(val_error)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}")

def k_mean_dba_magnitude_test(x_test: np.ndarray, y_test: np.ndarray, model_path: str, n_clusters: int = N_CLUSTERS, plots: bool = False) -> np.ndarray:
    """
    Load a trained DBA k-means model from file and test it on new data.

    Args:
        x_test (np.ndarray): Test data features.
        y_test (np.ndarray): Test labels.
        model_path (str): Path to the trained model.
        n_clusters (int, optional): Number of clusters for evaluation. Defaults to N_CLUSTERS.
        plots (bool, optional): Whether to generate and save plots. Defaults to False.

    Returns:
        np.ndarray: Cluster labels predicted for the testing dataset.

    Raises:
        ValueError: If clustering fails.
        FileNotFoundError: If the specified model path does not exist.
    """

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file {model_path} does not exist.")

        print("\nStarting clustering testing")

        print(f"Loading model from: {model_path}clustering_model.joblib")
        model = joblib.load(f"{model_path}clustering_model.joblib")
        print("Model loaded successfully")

        y_pred_kmeans_test = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred_kmeans_test = model.predict(x_test)

        if plots:
            plot_traffic_type_distribution(y_test, y_pred_kmeans_test, n_clusters, f'{PLOTS_PATH}traffic_type_distribution_test.pdf')
            plot_silhouette(model, x_test, f'{PLOTS_PATH}silhouette_test.pdf')
        
        metrics(x_test, y_pred_kmeans_test, y_pred_kmeans_test, y_test)
        return y_pred_kmeans_test

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        raise
    except ValueError as val_error:
        print(val_error)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
def process_file(file: str, x_df: pd.DataFrame, columns: List[str], model: BaseEstimator) -> Tuple[str, int]:
    """
    Process a single file by aggregating specified columns in the dataframe and predicting using a model.
    
    Args:
        file (str): The name of the file to process.
        x_df (pd.DataFrame): The DataFrame containing data with a 'File' column indicating file names.
        columns (List[str]): List of columns to aggregate.
        model (BaseEstimator): Trained model with a predict method.

    Returns:
        Tuple[str, int]: A tuple containing the file name and the predicted label.
    """
    x_tmp = aggregate_columns(x_df.loc[x_df['File'] == file], columns)['Magnitude'].to_list()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_tmp = model.predict(x_tmp)
    return file, y_tmp[0]

def parallel_predictions(x_df: pd.DataFrame, files: List[str], columns: List[str], model: BaseEstimator) -> Tuple[List[int], Dict[str, int]]:
    """
    Perform parallel predictions on a list of files using the provided model.
    
    Args:
        x_df (pd.DataFrame): The DataFrame containing data with a 'File' column indicating file names.
        files (List[str]): List of file names to process.
        columns (List[str]): List of columns to aggregate.
        model (BaseEstimator): Trained model with a predict method.

    Returns:
        Tuple[List[int], Dict[str, int]]: A tuple with two elements:
            - A list of predicted labels for each file in the order of files.
            - A dictionary mapping each file name to its predicted label.
    """
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_file, [(file, x_df, columns, model) for file in files])
    
    y_pred_kmeans_test_dict = {file: y_tmp for file, y_tmp in results}
    y_pred_kmeans_test = [y_tmp for file, y_tmp in results]

    return y_pred_kmeans_test, y_pred_kmeans_test_dict

def k_mean_dba_magnitude_test_cross_validation(file_path: str, model_path: str, n_clusters: int = N_CLUSTERS, plots: bool = False) -> np.ndarray:
    """
    Load a trained DBA k-means model and perform cross-validation on test data.

    Args:
        file_path (str): Path to the CSV file with test data.
        model_path (str): Path to the trained model file.
        n_clusters (int, optional): Number of clusters. Defaults to N_CLUSTERS.
        plots (bool, optional): Whether to generate and save plots. Defaults to False.

    Returns:
        np.ndarray: Predicted cluster labels for the test dataset.

    Raises:
        ValueError: If clustering fails.
        FileNotFoundError: If the specified model file does not exist.
    """

    try:

        print("\nStarting cross validating CNN with Clustering")

        columns = ['File', 'Magnitude']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file {model_path} does not exist.")

        print(f"Loading model from: {model_path}")
        model = joblib.load(f"{model_path}clustering_model.joblib")
        print("Model loaded successfully")

        y_test = load_csv(file_path, ['File', 'Labels'])
        if y_test is None:
            raise ValueError("Failed to load the CSV file for y_test.")

        y_test = y_test.groupby('File').agg({'Labels': 'first'}).reset_index()['Labels'].to_list()

        x_df = load_csv(file_path, columns)
        x_df = x_df.sort_values(by='File').reset_index(drop=True)
        x_test = np.array(aggregate_columns(x_df, columns)['Magnitude'].to_list())
        files = x_df['File'].unique()
        
        '''
        y_pred_kmeans_test = []
        y_pred_kmeans_test_dict = {}

        for file in files:
            x_tmp = aggregate_columns(x_df.loc[x_df['File'] == file], columns)['Magnitude'].to_list()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_tmp = model.predict(x_tmp)
                y_pred_kmeans_test_dict[file] = y_tmp[0]
            y_pred_kmeans_test.append(y_tmp[0])
        '''

        y_pred_kmeans_test, y_pred_kmeans_test_dict = parallel_predictions(x_df, files, columns, model)

        if plots:
            plot_traffic_type_distribution(y_test, y_pred_kmeans_test, n_clusters, f'{PLOTS_PATH}traffic_type_distribution_test_crossvalidation.pdf')
            plot_silhouette(model, x_test, f'{PLOTS_PATH}silhouette_test_crossvalidation.pdf')

        report = defaultdict(list)

        y_test = load_csv(file_path, ['File', 'Labels'])

        # Gather predictions for each technology
        for key, value in y_pred_kmeans_test_dict.items():
#            print(key, "predicted: " + STATIC_LABELS[value], "actual: " +  STATIC_LABELS[y_test.loc[y_test['File'] == key, 'Labels'].iloc[0]])
            technology = STATIC_LABELS[y_test.loc[y_test['File'] == key, 'Labels'].iloc[0]]
            report[technology].append(value)

#        print(report)

        # Count occurrences for each technology
        report = {tech: {i: count for i, count in enumerate(np.bincount(values)) if count > 0} 
                for tech, values in report.items()}

        # Print the clustering report
        print("\nReport of clustering")
        for technology, counts in report.items():
            print(f'\t{technology}')
            for cluster_id, count in counts.items():
                print(f'\t\t{cluster_id} : {count}')

        #metrics(x_test, y_pred_kmeans_test, y_pred_kmeans_test, y_test)

        return np.array(y_pred_kmeans_test)

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        raise
    except ValueError as val_error:
        print(val_error)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}")