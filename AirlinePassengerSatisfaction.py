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
from sklearn.decomposition import PCA

def adverserial_rf(data):
    adv_rf = arf.arf(x = data)
    
    # Get density estimates
    adv_rf.forde()

    # Generate data
    data_syn = adv_rf.forge(n = len(data)*1)
    
    return data_syn

#Function to plot original data distribution vs synthetic data distribution
def syn_test(original_data, synthetic_data):
    for column in original_data.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(original_data[column], label='Original', kde=True, color='blue', stat='density', bins=30)
        sns.histplot(synthetic_data[column], label='Synthetic', kde=True, color='red', stat='density', bins=30)
        plt.legend()
        plt.title(f'Distribution of {column}')
        plt.show()

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
       
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto',batch_size=50000)
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

def feature_selection_algorithm(dataset, original_columns, optimal_k, B=20, bootstrap_iterations=20):
    results = []    
    
    for i in range(B):
        print(i)
        # Step 1: Split the dataset
        D1, D2 = train_test_split(dataset, test_size=0.5, random_state=i)  # Different random state for each iteration

        # Step 2: Clustering on D1
        #kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, n_init='auto', batch_size=50000)
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto',batch_size=50000)
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
train_data_path = "data/AirlinePassengerSatisfaction/AirlinePassengerSatisfaction_train.csv"
test_data_path = "data/AirlinePassengerSatisfaction/AirlinePassengerSatisfaction_test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

data = pd.concat([train_data, test_data], ignore_index=True)

print(f"Shape of the original dataset: {data.shape}")

data = data.drop(data.columns[[0, 1, -1]], axis=1)

# Handle categorical data
data = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)

data = data.dropna()

#################################################################################################

print(f"Shape of the original dataset after preprocessing: {data.shape}")

# Store original column names
original_columns = data.columns.tolist()

# Silhouette Score
silhouette_scores = []
for k in range(2, 11, 2):
    print(k)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto',batch_size=50000)
    #kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
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


data_syn = adverserial_rf(data_reduced)

syn_test(data_reduced, data_syn)

data_reduced_augmented = pd.concat([data_reduced, data_syn], axis=0, ignore_index=True)

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

#Plot Clusters 

plot_clusters(data, labels, "Clustering Results - Original Dataset")
plot_clusters(data_reduced, labels_reduced, "Clustering Results - Reduced Dataset")
plot_clusters(data_reduced_augmented, labels_reduced_augmented, "Clustering Results - Reduced Dataset with Synthetic Data")

"""
output :
    Shape of the original dataset: (129880, 25)
    Shape of the original dataset after preprocessing: (129487, 23)
    
    Optimal number of clusters: 2
    
    Top features: ['Flight_Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Checkin service', 'Departure/Arrival time convenient', 'Class_Eco', 'Leg room service', 'Ease of Online booking', 'Online boarding', 'Gate location', 'Food and drink']
    Shape of the reduced dataset: (129487, 12)
    
    KMeans Clustering Evaluation
    Original Dataset: Silhouette Score: 0.12242273919729742, DBI: 2.644058977618708, CHI: 17338.612018289587, k: 2
    Reduced Dataset: Silhouette Score: 0.12624358442522332, DBI: 2.5634840797826186, CHI: 17536.686238524308, k: 2
    Reduced Dataset with Synthetic Data: Silhouette Score: 0.12506836179184624, DBI: 2.275465005979932, CHI: 31402.38043372465, k: 3
"""