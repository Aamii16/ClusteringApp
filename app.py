from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_for_clustering(image_path):
    """Process image and extract features for clustering"""
    try:
        img = Image.open(image_path)
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image for faster processing
        img = img.resize((100, 100))
        
        # Convert to numpy array and reshape
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        
        return pixels, img_array.shape
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
        
        # Basic preprocessing for social media data
        # Remove non-numeric columns for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            # If no numeric columns, try to extract features from text
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                # Simple text feature extraction (length, word count, etc.)
                features = []
                for col in text_cols[:3]:  # Take first 3 text columns
                    if col in df.columns:
                        features.append(df[col].astype(str).str.len())
                        features.append(df[col].astype(str).str.split().str.len())
                
                if features:
                    feature_df = pd.concat(features, axis=1)
                    feature_df.columns = [f'feature_{i}' for i in range(len(features))]
                    return feature_df.fillna(0).values, df
        
        # Use numeric columns
        data = df[numeric_cols].fillna(0).values
        return data, df
        
    except Exception as e:
        raise Exception(f"Error processing social media data: {str(e)}")

def perform_clustering(data, algorithm, n_clusters, random_state=42):
    """Perform clustering using specified algorithm"""
    try:
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        elif algorithm == 'spectral':
            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
        else:
            raise Exception("Unsupported clustering algorithm")
        
        labels = clusterer.fit_predict(data_scaled)
        return labels, data_scaled
    except Exception as e:
        raise Exception(f"Error performing clustering: {str(e)}")

def create_visualization(data, labels, data_type, original_shape=None):
    """Create visualization for clustering results"""
    try:
        plt.style.use('default')
        
        if data_type == 'image':
            # For image clustering, show original image and clustered version
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            original_img = data.reshape(original_shape)
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Clustered image
            clustered_img = labels.reshape(original_shape[:2])
            im = axes[1].imshow(clustered_img, cmap='viridis')
            axes[1].set_title('Clustered Regions')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
        else:
            # For dataset clustering, use PCA for 2D visualization
            if data.shape[1] > 2:
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(data)
            else:
                data_2d = data[:, :2]
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
            plt.title('Clustering Results (PCA Visualization)')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
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
    try:
        # Get form data
        algorithm = request.form.get('algorithm', 'kmeans')
        n_clusters = int(request.form.get('n_clusters', 3))
        data_type = request.form.get('data_type', 'image')
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            if data_type == 'image':
                # Process image
                data, original_shape = process_image_for_clustering(file_path)
                
                # Perform clustering
                labels, data_scaled = perform_clustering(data, algorithm, n_clusters)
                
                # Create visualization
                viz_base64 = create_visualization(data, labels, 'image', original_shape)
                
                # Calculate cluster statistics
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_stats = {f'Cluster {i}': int(count) for i, count in zip(unique_labels, counts)}
                
            else:
                # Process social media dataset
                file_ext = filename.split('.')[-1].lower()
                data, original_df = process_social_media_data(file_path, file_ext)
                
                # Perform clustering
                labels, data_scaled = perform_clustering(data, algorithm, n_clusters)
                
                # Create visualization
                viz_base64 = create_visualization(data_scaled, labels, 'dataset')
                
                # Calculate cluster statistics
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_stats = {f'Cluster {i}': int(count) for i, count in zip(unique_labels, counts)}
                
                # Add cluster labels to original dataframe for download
                result_df = original_df.copy()
                result_df['Cluster'] = labels
                
                # Save clustered dataset
                result_filename = f"clustered_{filename.split('.')[0]}.csv"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                result_df.to_csv(result_path, index=False)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'visualization': viz_base64,
                'cluster_stats': cluster_stats,
                'algorithm_used': algorithm.upper(),
                'n_clusters': n_clusters,
                'data_type': data_type,
                'download_available': data_type == 'dataset'
            })
            
        except Exception as e:
            # Clean up uploaded file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
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