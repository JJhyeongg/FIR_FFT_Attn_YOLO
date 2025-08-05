import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os

class SpatialFrequencyFilters:
    def __init__(self, filter_length=51):
        """
        Spatial frequency FIR filters for image processing
        
        Args:
            filter_length: Length of the FIR filter (should be odd)
        """
        self.filter_length = filter_length
        self.filters = self._design_filters()
    
    def _design_filters(self):
        """Design FIR filters for different frequency bands"""
        filters = {}
        
        # Normalized frequency ranges (0 to 1, where 1 is Nyquist frequency)
        # High pass: 2/3 to 1 (normalized)
        # Band pass: 1/3 to 2/3 (normalized)  
        # Low pass: 0 to 1/3 (normalized)
        
        # Design low pass filter (0 to 1/3)
        cutoff_low = 1/3
        filters['low_pass'] = signal.firwin(
            self.filter_length, 
            cutoff_low, 
            window='hamming'
        )
        
        # Design band pass filter (1/3 to 2/3)
        low_cutoff = 1/3
        high_cutoff = 2/3
        filters['band_pass'] = signal.firwin(
            self.filter_length,
            [low_cutoff, high_cutoff],
            window='hamming',
            pass_zero=False
        )
        
        # Design high pass filter (2/3 to 1)
        cutoff_high = 2/3
        filters['high_pass'] = signal.firwin(
            self.filter_length,
            cutoff_high,
            window='hamming',
            pass_zero=False
        )
        
        return filters
    
    def apply_2d_filter(self, image, filter_1d):
        """
        Apply 1D filter to 2D image using separable filtering
        
        Args:
            image: 2D numpy array (grayscale image)
            filter_1d: 1D FIR filter coefficients
            
        Returns:
            filtered_image: 2D numpy array
        """
        # Apply filter horizontally
        filtered_h = signal.convolve2d(image, filter_1d.reshape(1, -1), 
                                     mode='same', boundary='symm')
        
        # Apply filter vertically
        filtered_2d = signal.convolve2d(filtered_h, filter_1d.reshape(-1, 1), 
                                      mode='same', boundary='symm')
        
        return filtered_2d
    
    def process_rgb_image(self, rgb_image):
        """
        Process RGB image and generate frequency-separated images
        
        Args:
            rgb_image: numpy array of shape (height, width, 3) with RGB channels
            
        Returns:
            results: dictionary containing filtered images for each frequency band
        """
        height, width, channels = rgb_image.shape
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Extract individual channels
        r_channel = rgb_image[:, :, 0]
        g_channel = rgb_image[:, :, 1]
        b_channel = rgb_image[:, :, 2]
        
        results = {
            'high_freq': {},
            'mid_freq': {},
            'low_freq': {}
        }
        
        # Process each image type with each filter
        images = {
            'grayscale': gray_image,
            'r_channel': r_channel,
            'g_channel': g_channel,
            'b_channel': b_channel
        }
        
        for img_type, img in images.items():
            # Convert to float for processing
            img_float = img.astype(np.float32)
            
            # Apply high pass filter
            high_filtered = self.apply_2d_filter(img_float, self.filters['high_pass'])
            results['high_freq'][img_type] = np.clip(high_filtered, 0, 255).astype(np.uint8)
            
            # Apply band pass filter
            band_filtered = self.apply_2d_filter(img_float, self.filters['band_pass'])
            results['mid_freq'][img_type] = np.clip(band_filtered, 0, 255).astype(np.uint8)
            
            # Apply low pass filter
            low_filtered = self.apply_2d_filter(img_float, self.filters['low_pass'])
            results['low_freq'][img_type] = np.clip(low_filtered, 0, 255).astype(np.uint8)
        
        return results
    
    def visualize_filters(self):
        """Visualize the frequency response of the designed filters"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot impulse responses
        for i, (name, filt) in enumerate(self.filters.items()):
            axes[0, i].plot(filt)
            axes[0, i].set_title(f'{name.replace("_", " ").title()} - Impulse Response')
            axes[0, i].set_xlabel('Sample')
            axes[0, i].set_ylabel('Amplitude')
            axes[0, i].grid(True)
        
        # Plot frequency responses
        for i, (name, filt) in enumerate(self.filters.items()):
            w, h = signal.freqz(filt, worN=8000)
            axes[1, i].plot(w/np.pi, 20*np.log10(np.abs(h)))
            axes[1, i].set_title(f'{name.replace("_", " ").title()} - Frequency Response')
            axes[1, i].set_xlabel('Normalized Frequency (×π rad/sample)')
            axes[1, i].set_ylabel('Magnitude (dB)')
            axes[1, i].grid(True)
            axes[1, i].set_ylim(-80, 5)
            
            # Add vertical lines for cutoff frequencies
            if name == 'low_pass':
                axes[1, i].axvline(x=1/3, color='r', linestyle='--', alpha=0.7, label='Cutoff')
            elif name == 'band_pass':
                axes[1, i].axvline(x=1/3, color='r', linestyle='--', alpha=0.7, label='Lower cutoff')
                axes[1, i].axvline(x=2/3, color='r', linestyle='--', alpha=0.7, label='Upper cutoff')
            elif name == 'high_pass':
                axes[1, i].axvline(x=2/3, color='r', linestyle='--', alpha=0.7, label='Cutoff')
            
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.show()
    
    def display_results(self, results, original_image):
        """Display the filtered results"""
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        # Display original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original RGB Image')
        axes[0, 0].axis('off')
        
        # Display original grayscale
        gray_orig = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        axes[0, 1].imshow(gray_orig, cmap='gray')
        axes[0, 1].set_title('Original Grayscale')
        axes[0, 1].axis('off')
        
        # Clear unused subplots in first row
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')
        
        # Display filtered results
        freq_names = ['High Frequency', 'Mid Frequency', 'Low Frequency']
        freq_keys = ['high_freq', 'mid_freq', 'low_freq']
        channel_names = ['Grayscale', 'R Channel', 'G Channel', 'B Channel']
        channel_keys = ['grayscale', 'r_channel', 'g_channel', 'b_channel']
        
        for i, (freq_name, freq_key) in enumerate(zip(freq_names, freq_keys)):
            for j, (ch_name, ch_key) in enumerate(zip(channel_names, channel_keys)):
                axes[i+1, j].imshow(results[freq_key][ch_key], cmap='gray')
                axes[i+1, j].set_title(f'{freq_name}\n{ch_name}')
                axes[i+1, j].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
def create_test_image():
    """Create a test image with different frequency patterns"""
    # Create a 640x640 RGB test image
    height, width = 640, 640
    
    # Create different frequency patterns
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # High frequency pattern
    high_freq_pattern = np.sin(8*X) * np.sin(8*Y)
    
    # Mid frequency pattern
    mid_freq_pattern = np.sin(4*X) * np.sin(4*Y)
    
    # Low frequency pattern
    low_freq_pattern = np.sin(X) * np.sin(Y)
    
    # Combine patterns for each channel
    r_channel = ((high_freq_pattern + mid_freq_pattern + low_freq_pattern + 3) / 6 * 255).astype(np.uint8)
    g_channel = ((mid_freq_pattern + low_freq_pattern + 2) / 4 * 255).astype(np.uint8)
    b_channel = ((low_freq_pattern + 1) / 2 * 255).astype(np.uint8)
    
    # Stack channels
    test_image = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    return test_image

class SpatialFrequencyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spatial Frequency Filter GUI")
        self.root.geometry("1400x900")
        
        # Initialize filters
        self.spatial_filters = SpatialFrequencyFilters(filter_length=51)
        self.current_image = None
        self.current_results = None
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File selection
        ttk.Button(control_frame, text="Select Image File", 
                  command=self.select_image_file).grid(row=0, column=0, padx=(0, 10))
        
        # Filter length setting
        ttk.Label(control_frame, text="Filter Length:").grid(row=0, column=1, padx=(0, 5))
        self.filter_length_var = tk.IntVar(value=51)
        filter_length_spinbox = ttk.Spinbox(control_frame, from_=21, to=101, increment=10,
                                          textvariable=self.filter_length_var, width=10)
        filter_length_spinbox.grid(row=0, column=2, padx=(0, 10))
        
        # Update filters button
        ttk.Button(control_frame, text="Update Filters", 
                  command=self.update_filters).grid(row=0, column=3, padx=(0, 10))
        
        # Process button
        self.process_button = ttk.Button(control_frame, text="Process Image", 
                                       command=self.process_image, state='disabled')
        self.process_button.grid(row=0, column=4, padx=(0, 10))
        
        # Save results button
        self.save_button = ttk.Button(control_frame, text="Save Results", 
                                    command=self.save_results, state='disabled')
        self.save_button.grid(row=0, column=5, padx=(0, 10))
        
        # Show filter response button
        ttk.Button(control_frame, text="Show Filter Response", 
                  command=self.show_filter_response).grid(row=0, column=6)
        
        # Image display area
        self.create_image_display_area(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select an image file to start")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def create_image_display_area(self, parent):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image tab
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="Original Image")
        
        # Create canvas for original image
        self.original_canvas = tk.Canvas(self.original_frame, width=640, height=640, bg='white')
        self.original_canvas.pack(expand=True, fill='both')
        
        # Results tabs
        self.result_frames = {}
        self.result_canvases = {}
        
        freq_bands = [('High Frequency', 'high_freq'), 
                     ('Mid Frequency', 'mid_freq'), 
                     ('Low Frequency', 'low_freq')]
        
        for band_name, band_key in freq_bands:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=band_name)
            self.result_frames[band_key] = frame
            
            # Create 2x2 grid for the 4 channel images
            canvas_frame = ttk.Frame(frame)
            canvas_frame.pack(expand=True, fill='both', padx=5, pady=5)
            
            # Configure grid
            canvas_frame.columnconfigure(0, weight=1)
            canvas_frame.columnconfigure(1, weight=1)
            canvas_frame.rowconfigure(0, weight=1)
            canvas_frame.rowconfigure(1, weight=1)
            
            canvases = {}
            channels = [('Grayscale', 'grayscale'), ('R Channel', 'r_channel'),
                       ('G Channel', 'g_channel'), ('B Channel', 'b_channel')]
            
            for i, (ch_name, ch_key) in enumerate(channels):
                row = i // 2
                col = i % 2
                
                # Create label frame
                label_frame = ttk.LabelFrame(canvas_frame, text=ch_name, padding="5")
                label_frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), 
                               padx=2, pady=2)
                
                # Create canvas
                canvas = tk.Canvas(label_frame, width=300, height=300, bg='white')
                canvas.pack(expand=True, fill='both')
                canvases[ch_key] = canvas
            
            self.result_canvases[band_key] = canvases
    
    def select_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.load_image(file_path)
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                self.process_button.config(state='normal')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")
    
    def load_image(self, file_path):
        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Could not load image file")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x640 if needed
        if image.shape[:2] != (640, 640):
            image = cv2.resize(image, (640, 640))
        
        self.current_image = image
        
        # Display original image
        self.display_image_on_canvas(self.original_canvas, image)
        
        # Clear previous results
        self.current_results = None
        self.clear_result_canvases()
        self.save_button.config(state='disabled')
    
    def display_image_on_canvas(self, canvas, image, max_size=None):
        if max_size is None:
            max_size = (canvas.winfo_width(), canvas.winfo_height())
            if max_size[0] <= 1 or max_size[1] <= 1:  # Canvas not yet rendered
                max_size = (300, 300)
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image)
        else:  # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        
        # Resize image to fit canvas while maintaining aspect ratio
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                          anchor=tk.CENTER, image=photo)
        
        # Keep reference to prevent garbage collection
        canvas.image = photo
    
    def clear_result_canvases(self):
        for band_key in self.result_canvases:
            for ch_key in self.result_canvases[band_key]:
                canvas = self.result_canvases[band_key][ch_key]
                canvas.delete("all")
                canvas.image = None
    
    def update_filters(self):
        filter_length = self.filter_length_var.get()
        if filter_length % 2 == 0:  # Ensure odd length
            filter_length += 1
            self.filter_length_var.set(filter_length)
        
        self.spatial_filters = SpatialFrequencyFilters(filter_length=filter_length)
        self.status_var.set(f"Filters updated with length {filter_length}")
    
    def process_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.status_var.set("Processing image...")
            self.root.update()
            
            # Process the image
            self.current_results = self.spatial_filters.process_rgb_image(self.current_image)
            
            # Display results
            self.display_results()
            
            self.status_var.set("Image processing completed")
            self.save_button.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_var.set("Error processing image")
    
    def display_results(self):
        if self.current_results is None:
            return
        
        freq_bands = ['high_freq', 'mid_freq', 'low_freq']
        channels = ['grayscale', 'r_channel', 'g_channel', 'b_channel']
        
        for band_key in freq_bands:
            for ch_key in channels:
                canvas = self.result_canvases[band_key][ch_key]
                image = self.current_results[band_key][ch_key]
                
                # Update canvas after it's been rendered
                self.root.update_idletasks()
                self.display_image_on_canvas(canvas, image, max_size=(290, 290))
    
    def save_results(self):
        if self.current_results is None:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
        
        try:
            freq_bands = [('high_freq', 'High_Frequency'), 
                         ('mid_freq', 'Mid_Frequency'), 
                         ('low_freq', 'Low_Frequency')]
            channels = [('grayscale', 'Grayscale'), ('r_channel', 'R_Channel'),
                       ('g_channel', 'G_Channel'), ('b_channel', 'B_Channel')]
            
            saved_files = []
            for band_key, band_name in freq_bands:
                for ch_key, ch_name in channels:
                    image = self.current_results[band_key][ch_key]
                    filename = f"{band_name}_{ch_name}.png"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save using PIL
                    pil_image = Image.fromarray(image, mode='L')
                    pil_image.save(filepath)
                    saved_files.append(filename)
            
            messagebox.showinfo("Success", f"Saved {len(saved_files)} images to:\n{save_dir}")
            self.status_var.set(f"Results saved to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def show_filter_response(self):
        # Show filter response in a new window
        plt.figure(figsize=(15, 10))
        self.spatial_filters.visualize_filters()
        plt.show()

# GUI Application class
class SpatialFrequencyApp:
    def __init__(self):
        self.root = tk.Tk()
        self.gui = SpatialFrequencyGUI(self.root)
    
    def run(self):
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    # Create spatial frequency filters
    spatial_filters = SpatialFrequencyFilters(filter_length=51)
    
    # Choice: GUI or command line
    choice = input("Choose mode:\n1. GUI Mode\n2. Command Line Mode (with test image)\n3. Command Line Mode (with your image)\nEnter choice (1/2/3): ")
    
    if choice == "1":
        # Run GUI application
        app = SpatialFrequencyApp()
        app.run()
    elif choice == "2":
        # Command line with test image
        print("Running command line mode with test image...")
        # Visualize filter characteristics
        print("Filter characteristics:")
        spatial_filters.visualize_filters()
        
        # Create test image
        test_image = create_test_image()
        
        # Process the image
        print("Processing image...")
        results = spatial_filters.process_rgb_image(test_image)
        
        # Display results
        print("Displaying results...")
        spatial_filters.display_results(results, test_image)
    elif choice == "3":
        # Command line with user image
        image_path = input("Enter image file path: ")
        if os.path.exists(image_path):
            results = process_your_image(image_path)
        else:
            print("Image file not found!")
    else:
        print("Invalid choice. Running GUI mode by default.")
        app = SpatialFrequencyApp()
        app.run()