import nd2
import numpy as np
from skimage.transform import resize

class ND2Handler:
    def __init__(self, file_path):
        self.file_path = file_path
        self._pixel_size = None
        
    @property
    def pixel_size_um(self):
        """Get pixel size in micrometers"""
        if self._pixel_size is None:
            with nd2.ND2File(self.file_path) as nd2_file:
                # Try different metadata locations for pixel size
                try:
                    # First try to get from pixel calibration
                    self._pixel_size = nd2_file.metadata.channels[0].volume.axesCalibration[0]
                except:
                    try:
                        # Alternative: try to get from voxel size
                        self._pixel_size = nd2_file.voxel_size().x
                    except:
                        print("Warning: Could not get pixel size from metadata, using default value")
                        self._pixel_size = 0.325  # Default value
                
                print(f"Pixel size: {self._pixel_size:.3f} µm")
        
        return self._pixel_size
        
    def get_available_channels(self):
        """
        Get list of available channels in the ND2 file
        """
        with nd2.ND2File(self.file_path) as nd2_file:
            # Get channel information
            try:
                # Try to get channel names if available
                return [ch.channel.name for ch in nd2_file.metadata.channels]
            except:
                # If names not available, return channel indices
                return [f"Channel_{i}" for i in range(len(nd2_file.metadata.channels))]
        
    def read_channels(self, dapi_channel=None, red_channel=None):
        """
        Read specific channels from ND2 file
        Returns:
            Tuple of (red_channel, dapi_channel) to maintain compatibility
        """
        with nd2.ND2File(self.file_path) as nd2_file:
            # Get number of channels
            n_channels = len(nd2_file.metadata.channels)
            print(f"Number of channels: {n_channels}")
            
            # First channel (0) is DAPI/nuclei, Second channel (1) is red/cell
            dapi_idx = 0  # DAPI/nuclei is first channel
            red_idx = 1   # Red/cell is second channel
            
            # Read the full image data
            images = nd2_file.asarray()
            
            # Extract channels
            dapi_image = images[dapi_idx]
            red_image = images[red_idx]
            
            print(f"Using channel {dapi_idx} for DAPI/nuclei")
            print(f"Using channel {red_idx} for Cell staining")
            
            # Return in order (red, dapi) to maintain compatibility
            return red_image, dapi_image
    
    def get_test_crop(self, image, crop_size=(512, 512), position='center', seed=None):
        """
        Crop a portion of the image for testing
        
        Args:
            image: Input image array
            crop_size: Tuple of (height, width) for crop size
            position: 'center' or 'random'
            seed: Random seed for reproducibility
        """
        h, w = image.shape
        ch, cw = crop_size
        
        if position == 'center':
            start_h = (h - ch) // 2
            start_w = (w - cw) // 2
        else:  # random
            if seed is not None:
                np.random.seed(seed)
            start_h = np.random.randint(0, h - ch)
            start_w = np.random.randint(0, w - cw)
            
        return image[start_h:start_h + ch, start_w:start_w + cw] 