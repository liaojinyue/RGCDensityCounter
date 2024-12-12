import napari
import numpy as np
from pathlib import Path
import argparse
import json
from qtpy.QtCore import Qt
from nd2handler import ND2Handler
import logging
import warnings

# Suppress OpenGL warning
warnings.filterwarnings('ignore', category=UserWarning, message='No OpenGL_accelerate module loaded')

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class HoverROISelector:
    def __init__(self, viewer, image):
        self.viewer = viewer
        self.image = image
        self.drawing = False
        self.points = []
        self.start_point = None
        self.completion_radius = 10
        self.roi_count = 0
        self.current_roi_points = []  # Track points of current ROI
        
        # Add image layer
        self.image_layer = viewer.add_image(
            image, 
            name='Image',
            contrast_limits=[np.percentile(image, 1), np.percentile(image, 99)]
        )
        
        # Create layers for visualization
        self.shapes_layer = viewer.add_shapes(
            name='ROIs',
            edge_width=2,
            edge_color='red',
            face_color='transparent'
        )
        
        # Add start point marker
        self.points_layer = viewer.add_points(
            name='Start Point',
            size=self.completion_radius*2,
            face_color='red',
            edge_color='white',
            symbol='o'
        )
        
        # Add progress points layer
        self.progress_layer = viewer.add_points(
            name='Progress',
            size=3,
            face_color='yellow',
            edge_color='white',
            symbol='o'
        )
        
        # Debug print to understand event handling
        print("Viewer attributes:", dir(viewer))
        
        # Attempt to connect events based on available methods
        try:
            # Try newer event system first
            viewer.mouse_press_callbacks.append(self._on_mouse_press)
            viewer.mouse_move_callbacks.append(self._on_mouse_move)
        except AttributeError:
            try:
                # Fallback to older method
                viewer.mouse_click_callbacks.append(self._on_click)
                viewer.mouse_drag_callbacks.append(self._on_move)
            except AttributeError:
                # Last resort: print warning
                print("WARNING: Could not attach mouse event callbacks")
        
        # Bind keys for additional functionality
        try:
            viewer.bind_key('z', self._undo_last_roi)
            viewer.bind_key('d', self._delete_roi)
        except Exception as e:
            print(f"WARNING: Could not bind keys: {e}")
        
        # Add help text
        try:
            viewer.text_overlay.visible = True
            viewer.text_overlay.text = (
                "ROI Selection Controls:\n"
                "Right-click: Start new ROI\n"
                "Move mouse: Draw ROI boundary\n"
                "Return to red circle: Complete ROI\n"
                "Press 'z': Undo last ROI\n"
                "Press 'd': Delete ROI under cursor\n"
                "Press 'q': Finish and save"
            )
        except Exception as e:
            print(f"WARNING: Could not set text overlay: {e}")
    
    def _on_mouse_press(self, event):
        # Check for right mouse button press
        if event.button == 2:  # Right mouse button
            self._start_roi(event.position)
    
    def _on_click(self, viewer, event):
        if event.button == Qt.RightButton:
            self._start_roi([event.position[0], event.position[1]])
    
    def _start_roi(self, position):
        if not self.drawing:
            self.drawing = True
            self.start_point = position
            self.points = [self.start_point]
            self.current_roi_points = [self.start_point]
            self.points_layer.data = [self.start_point]
            logging.info(f"Started ROI #{self.roi_count + 1}")
    
    def _on_mouse_move(self, event):
        if self.drawing:
            self._update_roi_drawing(event.position)
    
    def _on_move(self, viewer, event):
        if self.drawing:
            self._update_roi_drawing([event.position[0], event.position[1]])
    
    def _update_roi_drawing(self, current_point):
        self.points.append(current_point)
        self.current_roi_points.append(current_point)
        
        # Update progress visualization
        self.progress_layer.data = np.array(self.current_roi_points)
        
        if self._near_start(current_point):
            self._complete_roi()
        else:
            self.shapes_layer.data = [np.array(self.points)]
    
    def _near_start(self, current_point):
        if self.start_point is None:
            return False
        dist = np.sqrt(
            (current_point[0] - self.start_point[0])**2 +
            (current_point[1] - self.start_point[1])**2
        )
        return dist < self.completion_radius
    
    def _complete_roi(self):
        if len(self.points) < 3:
            logging.warning("ROI too small, cancelled")
            self._reset_roi_drawing()
            return
            
        self.drawing = False
        self.roi_count += 1
        
        # Add completed ROI to shapes layer
        self.shapes_layer.add(
            np.array(self.points),
            shape_type='polygon',
            edge_color='red',
            face_color='transparent',
            name=f'ROI_{self.roi_count}'  # Name ROIs for easier deletion
        )
        
        self._reset_roi_drawing()
        
        logging.info(f"Completed ROI #{self.roi_count}")
    
    def _reset_roi_drawing(self):
        # Clear temporary points
        self.points = []
        self.current_roi_points = []
        self.start_point = None
        self.points_layer.data = []
        self.progress_layer.data = []
    
    def _undo_last_roi(self, event=None):
        """Undo the last created ROI"""
        if self.shapes_layer.nshapes > 0:
            self.shapes_layer.selected_data = {self.shapes_layer.nshapes - 1}
            self.shapes_layer.remove_selected()
            self.roi_count -= 1
            logging.info(f"Undid last ROI (remaining: {self.roi_count})")
    
    def _delete_roi(self, event=None):
        """Delete ROI under cursor"""
        if self.shapes_layer.nshapes > 0:
            # Get ROI under cursor
            try:
                selected = self.shapes_layer.get_value(
                    self.viewer.cursor.position,
                    view_direction=self.viewer.camera.view_direction
                )
                if selected is not None:
                    self.shapes_layer.selected_data = {selected}
                    self.shapes_layer.remove_selected()
                    self.roi_count -= 1
                    logging.info(f"Deleted ROI (remaining: {self.roi_count})")
            except Exception as e:
                print(f"Error deleting ROI: {e}")
    
    def get_masks(self):
        """Convert ROIs to binary masks"""
        if not self.shapes_layer.data:
            return None
            
        # Create empty mask with same dimensions as image
        h, w = self.image.shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add each ROI to mask with unique ID
        for i, shape in enumerate(self.shapes_layer.data, 1):
            mask = self.shapes_layer.to_masks([shape])[0]
            combined_mask[mask] = i
            
        return combined_mask



def select_rois_for_nd2(nd2_path, channel='red', output_dir=None):
    """
    Select ROIs from an ND2 file using napari
    
    Args:
        nd2_path: Path to ND2 file
        channel: Which channel to use ('red' or 'dapi')
        output_dir: Directory to save results (default: same as nd2 file)
    """
    nd2_path = Path(nd2_path)
    if output_dir is None:
        output_dir = nd2_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load ND2 file
        logging.info(f"Loading ND2 file: {nd2_path}")
        handler = ND2Handler(str(nd2_path))
        red_channel, dapi_channel = handler.read_channels()
        
        # Select channel for ROI selection
        image = red_channel if channel == 'red' else dapi_channel
        
        # Create viewer and selector
        viewer = napari.Viewer()
        roi_selector = HoverROISelector(viewer, image)
        
        # Start the viewer
        napari.run()
        
        # Get masks after viewer is closed
        masks = roi_selector.get_masks()
        if masks is None:
            logging.warning("No ROIs were selected")
            return
        
        # Save masks
        mask_file = output_dir / f"{nd2_path.stem}_rois.npy"
        np.save(mask_file, masks)
        
        # Save metadata
        metadata = {
            'source_file': str(nd2_path),
            'channel_used': channel,
            'num_rois': roi_selector.roi_count,
            'pixel_size_um': handler.pixel_size_um,
            'mask_file': str(mask_file),
            'image_shape': image.shape
        }
        
        metadata_file = output_dir / f"{nd2_path.stem}_rois_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Saved {roi_selector.roi_count} ROIs to {mask_file}")
        logging.info(f"Saved metadata to {metadata_file}")
        
    except Exception as e:
        logging.error(f"Error processing {nd2_path}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Select ROIs from ND2 file using hover-based selection')
    parser.add_argument('nd2_path', type=str, help='Path to ND2 file')
    parser.add_argument('--channel', type=str, choices=['red', 'dapi'], default='red',
                      help='Channel to use for ROI selection (default: red)')
    parser.add_argument('--output', type=str, help='Output directory (default: same as nd2 file)')
    
    args = parser.parse_args()
    select_rois_for_nd2(args.nd2_path, args.channel, args.output)

if __name__ == '__main__':
    main()