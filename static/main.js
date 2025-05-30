// Main Application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Navigation and Tab Switching
    const navLinks = document.querySelectorAll('nav ul li a');
    const sections = document.querySelectorAll('.content-section');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all nav items
            document.querySelectorAll('nav ul li').forEach(item => {
                item.classList.remove('active');
            });
            
            // Add active class to clicked nav item
            this.parentElement.classList.add('active');
            
            // Get the target section id
            const targetId = this.getAttribute('href').substring(1);
            
            // Hide all sections
            sections.forEach(section => {
                section.classList.remove('active');
            });
            
            // Show target section
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Initialize Visualization Tabs
    const vizButtons = document.querySelectorAll('.viz-btn');
    
    vizButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons in this viz group
            const parentGroup = this.closest('.viz-buttons');
            parentGroup.querySelectorAll('.viz-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Here you would switch the visualization type
            // This would be connected to your visualization library
            const vizType = this.getAttribute('data-viz');
            console.log(`Switching to ${vizType} visualization`);
            
            // For demo purposes, we'll update the placeholder text
            const vizContainer = this.closest('.visualization-panel').querySelector('.placeholder-viz p');
            vizContainer.textContent = `${vizType.charAt(0).toUpperCase() + vizType.slice(1)} visualization will appear here`;
        });
    });

    // New Project Button Actions
    const newSocialBtn = document.getElementById('new-social');
    const newImageBtn = document.getElementById('new-image');
    
    if (newSocialBtn) {
        newSocialBtn.addEventListener('click', function() {
            // Switch to social media analysis tab
            document.querySelector('a[href="#social-analysis"]').click();
        });
    }
    
    if (newImageBtn) {
        newImageBtn.addEventListener('click', function() {
            // Switch to image analysis tab
            document.querySelector('a[href="#image-analysis"]').click();
        });
    }

    // File Upload Handlers
    const socialDataUpload = document.getElementById('social-data-upload');
    const imageUpload = document.getElementById('image-upload');
    const uploadedImagesText = document.querySelector('.uploaded-images p');
    
    if (socialDataUpload) {
        socialDataUpload.addEventListener('change', function(e) {
            if (this.files.length > 0) {
                console.log(`Selected file: ${this.files[0].name}`);
                // Add code to handle the dataset upload
            }
        });
    }
    
    if (imageUpload) {
        imageUpload.addEventListener('change', function(e) {
            const count = this.files.length;
            uploadedImagesText.textContent = `${count} image${count !== 1 ? 's' : ''} selected`;
            
            // Here you would typically process the images and show thumbnails
            console.log(`Selected ${count} images`);
        });
    }

    // Run Button Actions
    const runButtons = document.querySelectorAll('.run-btn');
    
    runButtons.forEach(button => {
        button.addEventListener('click', function() {
            const panel = this.closest('.input-panel');
            const algorithm = panel.querySelector('input[type="radio"]:checked').id;
            const isKmeans = algorithm.includes('kmeans');
            
            // Show "running" state
            this.textContent = 'Running...';
            this.disabled = true;
            
            // Simulate clustering process with timeout
            setTimeout(() => {
                this.textContent = 'Run Clustering';
                this.disabled = false;
                
                // Update metrics with random values for demo
                const metricsContainer = this.closest('.analysis-container').querySelector('.metrics');
                const metricValues = metricsContainer.querySelectorAll('.value');
                
                metricValues[0].textContent = (Math.random() * 0.5 + 0.5).toFixed(3); // Silhouette
                metricValues[1].textContent = isKmeans ? 
                    (Math.random() * 1000 + 500).toFixed(1) : // Inertia for K-means
                    (Math.random() * 0.5 + 0.3).toFixed(3);  // Modularity for Spectral
                metricValues[2].textContent = `${(Math.random() * 5 + 1).toFixed(2)}s`; // Time
                
                // In a real app, this is where you would update the visualization
                showDummyVisualization();
                
                console.log(`Running ${algorithm} clustering`);
            }, 2000);
        });
    });

    // Settings Toggles
    const darkModeToggle = document.getElementById('darkMode');
    
    if (darkModeToggle) {
        darkModeToggle.addEventListener('change', function() {
            if (this.checked) {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
        });
    }

    // Function to show dummy visualization (for demo purposes)
    function showDummyVisualization() {
        const activeSection = document.querySelector('.content-section.active');
        const vizArea = activeSection.querySelector('.viz-area');
        const placeholderDiv = vizArea.querySelector('.placeholder-viz');
        
        if (placeholderDiv) {
            // Remove placeholder and add dummy visualization
            placeholderDiv.innerHTML = '';
            
            if (activeSection.id === 'social-analysis') {
                // Create dummy network graph using SVG
                const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                svg.setAttribute('width', '100%');
                svg.setAttribute('height', '100%');
                svg.setAttribute('viewBox', '0 0 500 400');
                
                // Draw some random nodes and edges
                for (let i = 0; i < 30; i++) {
                    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    const x = Math.random() * 450 + 25;
                    const y = Math.random() * 350 + 25;
                    const clusterIndex = Math.floor(Math.random() * 5);
                    const colors = ['#4a6fff', '#ff4a6f', '#6fff4a', '#f4ff4a', '#bf4aff'];
                    
                    circle.setAttribute('cx', x);
                    circle.setAttribute('cy', y);
                    circle.setAttribute('r', Math.random() * 5 + 5);
                    circle.setAttribute('fill', colors[clusterIndex]);
                    
                    svg.appendChild(circle);
                    
                    // Draw some edges
                    if (i > 0) {
                        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                        const prevX = Math.random() * 450 + 25;
                        const prevY = Math.random() * 350 + 25;
                        
                        line.setAttribute('x1', prevX);
                        line.setAttribute('y1', prevY);
                        line.setAttribute('x2', x);
                        line.setAttribute('y2', y);
                        line.setAttribute('stroke', '#e2e8f0');
                        line.setAttribute('stroke-width', '1');
                        
                        svg.appendChild(line);
                    }
                }
                
                vizArea.appendChild(svg);
                
            } else if (activeSection.id === 'image-analysis') {
                // Create dummy image grid
                const grid = document.createElement('div');
                grid.style.display = 'grid';
                grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(100px, 1fr))';
                grid.style.gap = '10px';
                grid.style.padding = '20px';
                grid.style.width = '100%';
                grid.style.height = '100%';
                grid.style.overflow = 'auto';
                
                const colors = ['#4a6fff', '#ff4a6f', '#6fff4a', '#f4ff4a', '#bf4aff'];
                
                // Create color blocks to represent images
                for (let i = 0; i < 30; i++) {
                    const clusterIndex = Math.floor(Math.random() * 5);
                    const imageDiv = document.createElement('div');
                    
                    imageDiv.style.backgroundColor = colors[clusterIndex];
                    imageDiv.style.borderRadius = '8px';
                    imageDiv.style.height = '100px';
                    imageDiv.style.display = 'flex';
                    imageDiv.style.alignItems = 'center';
                    imageDiv.style.justifyContent = 'center';
                    imageDiv.style.color = 'white';
                    imageDiv.style.fontWeight = 'bold';
                    imageDiv.innerText = `C${clusterIndex + 1}`;
                    
                    grid.appendChild(imageDiv);
                }
                
                vizArea.appendChild(grid);
            }
        }
    }
});

// Dark mode detector
if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    // Check if there's a setting toggle before enabling dark mode
    const darkModeToggle = document.getElementById('darkMode');
    if (darkModeToggle) {
        darkModeToggle.checked = true;
        document.body.classList.add('dark-mode');
    }
}