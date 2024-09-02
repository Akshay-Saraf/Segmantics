import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from arfpy import arf
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA

def adverserial_rf(data, chi_test = "no"):
    adv_rf = arf.arf(x = data)
    
    # Get density estimates
    adv_rf.forde()

    # Generate data
    data_syn = adv_rf.forge(n = len(data)*3)
    
    if chi_test == "yes":
        print("Synthetic Data Chi Square Test")
        print(syn_test(data, data_syn))
    
    data = pd.concat([data, data_syn], axis=0, ignore_index=True)
    return data

#Function to plot original data distribution vs synthetic data distribution
def syn_test(original_data, synthetic_data):
    for column in original_data.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(original_data[column], label='Original', kde=True, color='blue', stat='density', bins=30)
        sns.histplot(synthetic_data[column], label='Synthetic', kde=True, color='red', stat='density', bins=30)
        plt.legend()
        plt.title(f'Distribution of {column}')
        plt.show()
        
    categorical_columns = original_data.columns
        
    results = {}
    for column in categorical_columns:
        contingency_table = pd.crosstab(original_data[column], synthetic_data[column])
        stat, p_value, _, _ = chi2_contingency(contingency_table)
        results[column] = {'Chi-Square Statistic': stat, 'p-value': p_value}
    return pd.DataFrame(results).T

def cluster(data, optimal_k):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    best_k = 0
    best_silhouette = -1
    best_davies_bouldin = float('inf')
    best_calinski_harabasz = -1
    best_labels = None
    
    for k in range(optimal_k-1, optimal_k+2): #optimal_k-1 to optimal_k+1
    
        print(k)
        if k == 1:
            continue
       
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data_scaled)
            
        silhouette_avg = silhouette_score(data_scaled, labels)
        davies_bouldin_avg = davies_bouldin_score(data_scaled, labels)
        calinski_harabasz_avg = calinski_harabasz_score(data_scaled, labels)
        
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_davies_bouldin = davies_bouldin_avg
            best_calinski_harabasz = calinski_harabasz_avg
            best_k = k
            best_labels = labels

    return best_silhouette, best_davies_bouldin, best_calinski_harabasz, best_k, best_labels

def cluster_DBSCAN(data, eps_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], min_samples_values=[3, 5, 7, 10, 15]):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    best_silhouette = -1
    best_davies_bouldin = float('inf')
    best_calinski_harabasz = -1
    best_eps = 0
    best_min_samples = 0
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data_scaled)
            
            # Check if we have more than one cluster and if not all points are noise
            if len(set(labels)) > 1 and -1 in set(labels):
                silhouette_avg = silhouette_score(data_scaled, labels)
                davies_bouldin_avg = davies_bouldin_score(data_scaled, labels)
                calinski_harabasz_avg = calinski_harabasz_score(data_scaled, labels)

                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_davies_bouldin = davies_bouldin_avg
                    best_calinski_harabasz = calinski_harabasz_avg
                    best_eps = eps
                    best_min_samples = min_samples

    return best_silhouette, best_davies_bouldin, best_calinski_harabasz, best_eps, best_min_samples

def plot_clusters(data, labels, title):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar()
    plt.show()
    
def plot_single_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    # Since there's only one feature, the data points are on the x-axis
    plt.scatter(data, np.zeros_like(data), c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('Annual Income ($)')
    plt.colorbar()
    plt.show()

def feature_selection_algorithm(dataset, original_columns, optimal_k, B=20, bootstrap_iterations=20):
    results = []    
    
    for i in range(B):
        print(i)
        # Step 1: Split the dataset
        D1, D2 = train_test_split(dataset, test_size=0.5, random_state=i)  # Different random state for each iteration

        # Step 2: Clustering on D1
        #kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, n_init='auto', batch_size=50000)
        kmeans = KMeans(n_clusters=optimal_k, random_state=i, n_init='auto')
        D1_labels = kmeans.fit_predict(D1)
        D2_labels = kmeans.predict(D2)

        # Step 3: Transform to Supervised Learning
        combined_data = np.vstack((D1, D2))
        combined_labels = np.hstack((D1_labels, D2_labels))
        
        if i == 0:
            # Combine data and labels into a single DataFrame with original feature names
            combined_df = pd.DataFrame(combined_data, columns=original_columns)
            combined_df['label'] = combined_labels

        # Step 5: Iterative Validation (Bootstrapping and MCCV)
        bootstrap_importance_scores = []

        for _ in range(bootstrap_iterations):
            # Resample the dataset with replacement to create a bootstrap sample
            bootstrap_data, bootstrap_labels = resample(combined_data, combined_labels, random_state=_)
            rf_model = RandomForestClassifier()
            rf_model.fit(bootstrap_data, bootstrap_labels)
            bootstrap_importance_scores.append(rf_model.feature_importances_)
        
        # Aggregate the bootstrap importance scores by averaging
        aggregated_scores = np.mean(bootstrap_importance_scores, axis=0)
        results.append(aggregated_scores)

    # Step 6: Aggregation of Results
    final_scores = np.mean(results, axis=0)
    
    return combined_df, final_scores

#################################################################################################

# Load data
data_path = "data/ShopCustomerData/ShopCustomerData.csv"
data = pd.read_csv(data_path)

print(f"Shape of the original dataset: {data.shape}")

# Drop original name columns
data = data.drop(data.columns[[0]], axis=1)

# Handle categorical data
data = pd.get_dummies(data, columns=['Gender', 'Profession'], drop_first=True)

data = data.dropna()

#################################################################################################

print(f"Shape of the original dataset after preprocessing: {data.shape}")

# Store original column names
original_columns = data.columns.tolist()

# Silhouette Score
silhouette_scores = []
for k in range(2, 11, 2):
    print(k)
    #kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', batch_size=50000)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data)
    silhouette_scores.append(silhouette_score(data, labels))

# Find the optimal number of clusters
optimal_k = np.argmax(silhouette_scores) + 2  # +2 because the range starts from 2
print(f"Optimal number of clusters: {optimal_k}")
     
data, feature_scores = feature_selection_algorithm(data, original_columns, optimal_k)

# Identify top features explaining 90% of the variance
sorted_indices = np.argsort(feature_scores)[::-1]
sorted_features = np.array(data.columns)[sorted_indices]
sorted_importances = feature_scores[sorted_indices]
cumulative_importance = np.cumsum(sorted_importances)
top_features = sorted_features[cumulative_importance <= 0.9]

# Check if we need to include one more feature to reach at least 90%
if cumulative_importance[len(top_features)] < 0.9:
    top_features = sorted_features[:len(top_features)+1]

selected_features = list(top_features)

print(f"Top features: {selected_features}")

# Create a new dataset with the selected features
data_reduced = data[selected_features]

print(f"Shape of the reduced dataset: {data_reduced.shape}")

data_reduced_augmented = adverserial_rf(data_reduced, chi_test = "yes")

# Perform clustering
data_score, data_db, data_ch, data_k, labels = cluster(data, optimal_k)
#data_augmented_score, data_augmented_db, data_augmented_ch, data_augmented_k = cluster(data_augmented,optimal_k)
data_reduced_score, data_reduced_db, data_reduced_ch, data_reduced_k, labels_reduced = cluster(data_reduced, optimal_k)
data_reduced_augmented_score, data_reduced_augmented_db, data_reduced_augmented_ch, data_reduced_augmented_k, labels_reduced_augmented = cluster(data_reduced_augmented, optimal_k)


print("KMeans Clustering Evaluation")
print(f"Original Dataset: Silhouette Score: {data_score}, DBI: {data_db}, CHI: {data_ch}, k: {data_k}")
#print(f"Original Dataset with Synthetic data: Silhouette Score: {data_augmented_score}, DBI: {data_augmented_db}, CHI: {data_augmented_ch}, k: {data_augmented_k}")
print(f"Reduced Dataset: Silhouette Score: {data_reduced_score}, DBI: {data_reduced_db}, CHI: {data_reduced_ch}, k: {data_reduced_k}")
print(f"Reduced Dataset with Synthetic Data: Silhouette Score: {data_reduced_augmented_score}, DBI: {data_reduced_augmented_db}, CHI: {data_reduced_augmented_ch}, k: {data_reduced_augmented_k}")


data_dbscan_score, data_dbscan_db, data_dbscan_ch, data_dbscan_eps, data_dbscan_min_samples = cluster_DBSCAN(data)
#data_augmented_dbscan_score, data_augmented_dbscan_db, data_augmented_dbscan_ch, data_augmented_dbscan_eps, data_augmented_dbscan_min_samples = cluster_DBSCAN(data_augmented)
data_reduced_dbscan_score, data_reduced_dbscan_db, data_reduced_dbscan_ch, data_reduced_dbscan_eps, data_reduced_dbscan_min_samples = cluster_DBSCAN(data_reduced)
data_reduced_augmented_dbscan_score, data_reduced_augmented_dbscan_db, data_reduced_augmented_dbscan_ch, data_reduced_augmented_dbscan_eps, data_reduced_augmented_dbscan_min_samples = cluster_DBSCAN(data_reduced_augmented)

print("\nDBSCAN Clustering Evaluation")
print(f"Original Dataset: Silhouette Score: {data_dbscan_score}, DBI: {data_dbscan_db}, CHI: {data_dbscan_ch}, eps: {data_dbscan_eps}, min_samples: {data_dbscan_min_samples}")
#print(f"Original Dataset with Synthetic data: Silhouette Score: {data_augmented_dbscan_score}, DBI: {data_augmented_dbscan_db}, CHI: {data_augmented_dbscan_ch}, eps: {data_augmented_dbscan_eps}, min_samples: {data_augmented_dbscan_min_samples}")
print(f"Reduced Dataset: Silhouette Score: {data_reduced_dbscan_score}, DBI: {data_reduced_dbscan_db}, CHI: {data_reduced_dbscan_ch}, eps: {data_reduced_dbscan_eps}, min_samples: {data_reduced_dbscan_min_samples}")
print(f"Reduced Dataset with Synthetic Data: Silhouette Score: {data_reduced_augmented_dbscan_score}, DBI: {data_reduced_augmented_dbscan_db}, CHI: {data_reduced_augmented_dbscan_ch}, eps: {data_reduced_augmented_dbscan_eps}, min_samples: {data_reduced_augmented_dbscan_min_samples}")

#Plot Clusters 

plot_clusters(data, labels, "Clustering Results - Original Dataset")
plot_single_clusters(data_reduced, labels_reduced, "Clustering Results - Reduced Dataset")
plot_single_clusters(data_reduced_augmented, labels_reduced_augmented, "Clustering Results - Reduced Dataset with Synthetic Data")

"""
output :
    Shape of the original dataset: (2000, 8)
    Shape of the original dataset after preprocessing: (2000, 14)
    Optimal number of clusters: 2
    Top features: ['Annual Income ($)']
    Shape of the reduced dataset: (2000, 1)
    
                       Chi-Square Statistic   p-value
    Annual Income ($)          2.095439e+06  0.529347
    
    KMeans Clustering Evaluation
    Original Dataset: Silhouette Score: 0.12413146096952767, DBI: 2.6487626013702394, CHI: 262.7988275717551, k: 2
    Reduced Dataset: Silhouette Score: 0.6181468285305379, DBI: 0.5071741969936334, CHI: 5250.193619537169, k: 2
    Reduced Dataset with Synthetic Data: Silhouette Score: 0.622165857760359, DBI: 0.5017792512219303, CHI: 21409.998912776722, k: 2
    
    DBSCAN Clustering Evaluation
    Original Dataset: Silhouette Score: -0.15417322885714035, DBI: 2.1243879207812286, CHI: 11.87659521067509, eps: 1.0, min_samples: 10
    Reduced Dataset: Silhouette Score: -1, DBI: inf, CHI: -1, eps: 0, min_samples: 0
    Reduced Dataset with Synthetic Data: Silhouette Score: -1, DBI: inf, CHI: -1, eps: 0, min_samples: 0
"""