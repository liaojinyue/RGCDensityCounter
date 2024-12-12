import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import colorsys
from pathlib import Path
import json

def downsample_image(image, factor=8):
    """
    Downsample image by averaging blocks of pixels
    Handles arbitrary image sizes by padding if necessary
    """
    if factor == 1:
        return image
        
    h, w = image.shape
    
    # Calculate new dimensions
    h_new = h // factor
    w_new = w // factor
    
    # Calculate padding needed
    pad_h = (h_new * factor) - h
    pad_w = (w_new * factor) - w
    
    if pad_h < 0:
        h_new = (h // factor) + 1
        pad_h = (h_new * factor) - h
    if pad_w < 0:
        w_new = (w // factor) + 1
        pad_w = (w_new * factor) - w
    
    # Pad image if necessary
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    try:
        return image.reshape(h_new, factor, w_new, factor).mean(axis=(1,3))
    except ValueError:
        # Fallback method: simple resizing
        from skimage.transform import resize
        return resize(image, (h_new, w_new), preserve_range=True).astype(image.dtype)

class ROISelector:
    def __init__(self, image, roi_size=512, title="Select ROIs", downsample_factor=8):
        """
        Interactive ROI selector for large images
        
        Args:
            image: 2D numpy array of the image
            roi_size: Size of ROI in original image space (default 512x512)
            title: Window title
            downsample_factor: Factor to downsample image for display
        """
        self.original_image = image
        self.original_shape = image.shape
        self.downsample_factor = downsample_factor
        self.roi_size = roi_size
        self.crop_size = roi_size // downsample_factor
        
        # Downsample image for display
        self.display_image = downsample_image(image, downsample_factor)
        
        self.rois = []  # Will store coordinates in original image space
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        
        # Display downsampled image
        self.ax.imshow(self.display_image, cmap='gray')
        self.ax.set_title(title)
        
        # Create rectangle selector
        self.rect_selector = RectangleSelector(
            self.ax, 
            self.on_select,
            props=dict(facecolor='none', alpha=0.3),
            interactive=True,
            button=[1],
            spancoords='pixels'
        )
        
        # Add instructions
        plt.figtext(0.1, 0.01, 
                   "Click to place ROI. Press 'a' to add more ROIs.\n" +
                   "Press 'd' to delete last ROI. Press Enter when done.", 
                   wrap=True)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def get_roi_color(self, index):
        """Generate distinct colors for ROIs"""
        hue = (index * 0.618033988749895) % 1
        return colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        
    def on_select(self, eclick, erelease):
        """Handle ROI selection"""
        if eclick.xdata is None:
            return
            
        # Get click coordinates in downsampled space
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        
        # Ensure we stay within bounds
        h, w = self.display_image.shape
        x1 = max(0, min(x1, w - self.crop_size))
        y1 = max(0, min(y1, h - self.crop_size))
        
        # Calculate coordinates in original image space
        x1_orig = x1 * self.downsample_factor
        y1_orig = y1 * self.downsample_factor
        
        # Ensure we don't exceed original image bounds
        x1_orig = min(x1_orig, self.original_shape[1] - self.roi_size)
        y1_orig = min(y1_orig, self.original_shape[0] - self.roi_size)
        x2_orig = x1_orig + self.roi_size
        y2_orig = y1_orig + self.roi_size
        
        # Store ROI coordinates
        self.rois.append({
            'index': len(self.rois) + 1,
            'coords': (y1_orig, y2_orig, x1_orig, x2_orig),
            'rect': None,
            'text': None
        })
        
        # Draw ROI rectangle
        color = self.get_roi_color(len(self.rois) - 1)
        rect = plt.Rectangle((x1, y1), self.crop_size, self.crop_size,
                           fill=False, edgecolor=color, linewidth=2)
        text = self.ax.text(x1+2, y1+5, f'ROI {len(self.rois)}', 
                          color=color, fontsize=12, weight='bold')
        
        self.rois[-1]['rect'] = rect
        self.rois[-1]['text'] = text
        self.ax.add_patch(rect)
        
        self.fig.canvas.draw_idle()
        
    def delete_last_roi(self):
        """Remove the last added ROI"""
        if self.rois:
            last_roi = self.rois.pop()
            last_roi['rect'].remove()
            last_roi['text'].remove()
            self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Handle key press events"""
        if event.key == 'enter':
            plt.close()
        elif event.key == 'd':
            self.delete_last_roi()
    
    def get_rois(self):
        """Show selector and return ROI coordinates"""
        plt.show()
        return [roi['coords'] for roi in self.rois]
    
    def save_rois(self, save_path):
        """Save ROI coordinates to JSON file"""
        roi_info = {
            'rois': [
                {
                    'index': i+1,
                    'x_start': int(roi['coords'][2]),
                    'y_start': int(roi['coords'][0]),
                    'width': self.roi_size,
                    'height': self.roi_size
                }
                for i, roi in enumerate(self.rois)
            ],
            'image_shape': self.original_shape,
            'roi_size': self.roi_size
        }
        
        save_path = Path(save_path)
        with open(save_path, 'w') as f:
            json.dump(roi_info, f, indent=4) 