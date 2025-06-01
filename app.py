from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import json
import requests
from datetime import datetime
import time
from collections import Counter
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_color_palette(image_array, n_colors=5):
    """Extract dominant colors from image"""
    try:
        pixels = image_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        return colors.tolist()
    except Exception as e:
        return []

def process_image_for_clustering(image_path):
    """Process image and extract features for clustering"""
    try:
        img = Image.open(image_path)
        original_img = img.copy()
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((100, 100))
        img_array = np.array(img)
        original_array = np.array(original_img.resize((100, 100)))
        pixels = img_array.reshape(-1, 3)
        
        return pixels, img_array.shape, original_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def process_social_media_data(file_path, file_type):
    """Process social media dataset for clustering"""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'json':
            df = pd.read_json(file_path)
        else:
            raise Exception("Unsupported file type")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                features = []
                for col in text_cols[:3]:
                    if col in df.columns:
                        features.append(df[col].astype(str).str.len())
                        features.append(df[col].astype(str).str.split().str.len())
                
                if features:
                    feature_df = pd.concat(features, axis=1)
                    feature_df.columns = [f'feature_{i}' for i in range(len(features))]
                    return feature_df.fillna(0).values, df
        
        data = df[numeric_cols].fillna(0).values
        return data, df
        
    except Exception as e:
        raise Exception(f"Error processing social media data: {str(e)}")

def generate_mock_social_data(platform, data_type, search_query, num_points=100):
    """Generate mock social media data for demo purposes"""
    np.random.seed(42)
    data = []
    keywords = search_query.split() if search_query else ['sample', 'data', 'social']
    
    for i in range(num_points):
        if data_type == 'posts':
            engagement = np.random.randint(1, 1000)
            likes = np.random.randint(0, engagement)
            shares = np.random.randint(0, engagement // 2)
            comments = np.random.randint(0, engagement // 3)
            
            data.append({
                'id': i,
                'text': f"Sample post about {np.random.choice(keywords)} #{np.random.choice(keywords)}",
                'engagement_score': engagement,
                'likes': likes,
                'shares': shares,
                'comments': comments,
                'platform': platform
            })
        elif data_type == 'users':
            followers = np.random.randint(10, 10000)
            following = np.random.randint(10, 2000)
            posts_count = np.random.randint(1, 500)
            
            data.append({
                'user_id': i,
                'followers': followers,
                'following': following,
                'posts_count': posts_count,
                'engagement_rate': np.random.uniform(0.01, 0.1),
                'platform': platform
            })
    
    return pd.DataFrame(data)

def analyze_clusters_social(df, labels):
    """Analyze social media clusters and extract insights"""
    clusters_info = []
    
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = df[cluster_mask]
        cluster_size = len(cluster_data)
        
        keywords = []
        text_cols = cluster_data.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            text_col = text_cols[0]
            if text_col in cluster_data.columns:
                all_text = ' '.join(cluster_data[text_col].astype(str))
                words = re.findall(r'\b\w+\b', all_text.lower())
                word_counts = Counter(words)
                keywords = [word for word, count in word_counts.most_common(5) 
                          if len(word) > 3 and word not in ['sample', 'post', 'about']]
        
        characteristics = []
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:
            mean_val = cluster_data[col].mean()
            overall_mean = df[col].mean()
            
            if mean_val > overall_mean * 1.2:
                characteristics.append(f"High {col.replace('_', ' ')}")
            elif mean_val < overall_mean * 0.8:
                characteristics.append(f"Low {col.replace('_', ' ')}")
        
        if not characteristics:
            characteristics = ["Moderate activity", "Balanced engagement"]
        
        clusters_info.append({
            'size': cluster_size,
            'keywords': keywords[:5] if keywords else ['social', 'media', 'data'],
            'characteristics': characteristics[:3]
        })
    
    return clusters_info

def perform_clustering(data, algorithm, n_clusters, random_state=42, **kwargs):
    """Perform clustering using specified algorithm"""
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        if algorithm == 'kmeans':
            max_iter = kwargs.get('max_iterations', 100)
            clusterer = KMeans(
                n_clusters=n_clusters, 
                random_state=random_state, 
                n_init=10,
                max_iter=max_iter
            )
        elif algorithm == 'spectral':
            gamma = kwargs.get('gamma', 1.0)
            clusterer = SpectralClustering(
                n_clusters=n_clusters, 
                random_state=random_state,
                gamma=gamma
            )
        else:
            raise Exception("Unsupported clustering algorithm")
        
        labels = clusterer.fit_predict(data_scaled)
        return labels, data_scaled
    except Exception as e:
        raise Exception(f"Error performing clustering: {str(e)}")

def create_visualization(data, labels, mode, original_shape=None, original_image=None):
    """Create visualization for clustering results"""
    try:
        plt.style.use('default')
        
        if mode == 'image':
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            if original_image is not None:
                axes[0].imshow(original_image)
                axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')
            
            clustered_img = labels.reshape(original_shape[:2])
            im = axes[1].imshow(clustered_img, cmap='viridis')
            axes[1].set_title('Clustered Regions', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
        else:
            if data.shape[1] > 2:
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(data)
            else:
                data_2d = data[:, :2]
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, 
                                cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, label='Cluster')
            plt.title('Clustering Results (PCA Visualization)', fontsize=16, fontweight='bold')
            plt.xlabel('First Principal Component', fontsize=12)
            plt.ylabel('Second Principal Component', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        raise Exception(f"Error creating visualization: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    start_time = time.time()
    
    try:
        mode = request.form.get('mode', 'image')
        algorithm = request.form.get('algorithm', 'kmeans')
        n_clusters = int(request.form.get('num_clusters', 3))
        random_state = int(request.form.get('random_state', 42))
        
        clustering_params = {}
        if algorithm == 'kmeans':
            clustering_params['max_iterations'] = int(request.form.get('max_iterations', 100))
        elif algorithm == 'spectral':
            clustering_params['gamma'] = float(request.form.get('gamma', 1.0))
        
        if n_clusters < 2 or n_clusters > 20:
            return jsonify({'error': 'Number of clusters must be between 2 and 20'}), 400
        
        if mode == 'image':
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Supported: PNG, JPG, GIF'}), 400
            
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                data, original_shape, original_image = process_image_for_clustering(file_path)
                labels, data_scaled = perform_clustering(data, algorithm, n_clusters, 
                                                       random_state, **clustering_params)
                
                viz_base64 = create_visualization(data, labels, 'image', 
                                                original_shape, original_image)
                
                original_buffer = io.BytesIO()
                Image.fromarray(original_image).save(original_buffer, format='PNG')
                original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
                
                segmented_img = labels.reshape(original_shape[:2])
                segmented_buffer = io.BytesIO()
                plt.figure(figsize=(6, 6))
                plt.imshow(segmented_img, cmap='viridis')
                plt.axis('off')
                plt.savefig(segmented_buffer, format='png', bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                segmented_base64 = base64.b64encode(segmented_buffer.getvalue()).decode()
                plt.close()
                
                color_palette = extract_color_palette(original_image, n_clusters)
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_stats = {f'Cluster {i}': int(count) for i, count in zip(unique_labels, counts)}
                processing_time = round(time.time() - start_time, 2)
                
                os.remove(file_path)
                
                return jsonify({
                    'success': True,
                    'mode': 'image',
                    'original_image': original_base64,
                    'segmented_image': segmented_base64,
                    'visualization': viz_base64,
                    'color_palette': color_palette,
                    'cluster_stats': cluster_stats,
                    'algorithm_used': algorithm.upper(),
                    'num_clusters': n_clusters,
                    'processing_time': processing_time,
                    'pixels_processed': len(data),
                    'download_available': False
                })
                
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
                
        else:  # Social media mode
            platform = request.form.get('platform', 'twitter')
            data_type = request.form.get('data_type', 'posts')
            search_query = request.form.get('search_query', '')
            
            if platform == 'upload':
                if 'file' not in request.files:
                    return jsonify({'error': 'No dataset file uploaded'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if not allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type. Supported: CSV, JSON'}), 400
                
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                try:
                    file_ext = filename.split('.')[-1].lower()
                    data, original_df = process_social_media_data(file_path, file_ext)
                    os.remove(file_path)
                except Exception as e:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise e
            else:
                if not search_query.strip():
                    return jsonify({'error': 'Search query is required for API data fetching'}), 400
                
                original_df = generate_mock_social_data(platform, data_type, search_query)
                numeric_cols = original_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data = original_df[numeric_cols].fillna(0).values
                else:
                    return jsonify({'error': 'No numeric features found for clustering'}), 400
            
            labels, data_scaled = perform_clustering(data, algorithm, n_clusters, 
                                                   random_state, **clustering_params)
            viz_base64 = create_visualization(data_scaled, labels, 'social')
            clusters_info = analyze_clusters_social(original_df, labels)
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_stats = {f'Cluster {i}': int(count) for i, count in zip(unique_labels, counts)}
            
            result_df = original_df.copy()
            result_df['Cluster'] = labels
            result_filename = f"clustered_social_data_{timestamp}.csv"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            result_df.to_csv(result_path, index=False)
            
            processing_time = round(time.time() - start_time, 2)
            
            return jsonify({
                'success': True,
                'mode': 'social',
                'visualization': viz_base64,
                'clusters': clusters_info,
                'cluster_stats': cluster_stats,
                'algorithm_used': algorithm.upper(),
                'num_clusters': n_clusters,
                'processing_time': processing_time,
                'total_data_points': len(original_df),
                'download_available': True,
                'download_filename': result_filename
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)