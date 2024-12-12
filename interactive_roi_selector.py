from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
                           QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFileDialog,
                           QGraphicsItem, QMessageBox, QLabel)
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPainterPath
import numpy as np
import sys
from pathlib import Path
from nd2handler import ND2Handler
import logging
from skimage.draw import polygon
import json
from scipy.interpolate import splprep, splev
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

logging.basicConfig(level=logging.INFO)

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Initialize group colors
        self.group_colors = plt.cm.tab20(np.linspace(0, 1, 12))  # Get 12 distinct colors
        self.current_group = "Temporal_Peripheral"  # Update default group
        
        # Setup view
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Store nd2_handler reference
        self.nd2_handler = None  # Will be set by MainWindow
        
        # ROI drawing state
        self.drawing = False
        self.current_path = None
        self.start_point = None
        self.current_points = []
        self.rois = []
        self.completion_radius = 20
        
        # Initialize pens with better widths
        self.drawing_pen = QPen(Qt.red, 2)  # Increased from 0.1 to 2
        self.completion_pen = QPen(Qt.green, 2)
        self.drawing_pen.setCosmetic(True)
        self.completion_pen.setCosmetic(True)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Enable mouse pan and zoom
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Add keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus
        
        # Add state for reference circle
        self.waiting_for_reference = True
        self.reference_circle = None
        self.circle_radius = 3000  # pixels
        
        # Initialize instruction label as None
        self.instruction_label = None
        
        # Initialize reference circles list
        self.reference_circles = []
        
        # Add instruction label
        self.setup_instruction_label()
        
        # Add parameters for auto-completion and smoothing
        self.completion_threshold = 20  # pixels
        self.smoothing_factor = 1.0  # Increased for smoother curves
        self.min_points_for_completion = 10  # minimum points before auto-completion
        self.interpolation_points = 500  # More points for smoother appearance
        
        # Add regions list
        self.regions = [
            'Temporal_Peripheral', 'Temporal_Middle', 'Temporal_Central',
            'Superior_Peripheral', 'Superior_Middle', 'Superior_Central', 
            'Inferior_Peripheral', 'Inferior_Middle', 'Inferior_Central',
            'Nasal_Peripheral', 'Nasal_Middle', 'Nasal_Central'
        ]
        
        # Define custom colors for anatomical regions
        self.region_colors = {
            'Temporal_Peripheral': '#E76F51',  # Terracotta
            'Temporal_Middle': '#F4A261',      # Sandy brown
            'Temporal_Central': '#E67E22',     # Darker coral
            
            'Superior_Peripheral': '#1D3557',  # Dark blue
            'Superior_Middle': '#457B9D',      # Steel blue
            'Superior_Central': '#2E86C1',     # Darker blue
            
            'Inferior_Peripheral': '#2D6A4F',  # Forest green
            'Inferior_Middle': '#40916C',      # Medium green
            'Inferior_Central': '#2E7D32',     # Darker green
            
            'Nasal_Peripheral': '#7B2CBF',     # Deep purple
            'Nasal_Middle': '#9D4EDD',         # Medium purple
            'Nasal_Central': '#8E44AD'         # Darker purple
        }
        
        # Enhance rendering quality
        self.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing
        )

    def setup_instruction_label(self):
        """Create or update the instruction label"""
        try:
            # Remove old label if it exists
            if hasattr(self, 'instruction_label') and self.instruction_label is not None:
                try:
                    if self.instruction_label.scene() == self.scene:
                        self.scene.removeItem(self.instruction_label)
                except:
                    pass
                self.instruction_label = None
            
            # Create new label
            self.instruction_label = self.scene.addText("")
            self.instruction_label.setDefaultTextColor(Qt.white)
            font = self.instruction_label.font()
            font.setPointSize(14)
            font.setBold(True)
            self.instruction_label.setFont(font)
            self.instruction_label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            
        except Exception as e:
            logging.error(f"Error setting up instruction label: {str(e)}")

    def update_instructions(self, text):
        """Update floating instruction text"""
        try:
            if not self.scene:
                return
                
            # Create new label if needed
            if not hasattr(self, 'instruction_label') or self.instruction_label is None:
                self.setup_instruction_label()
            
            if self.instruction_label and self.instruction_label.scene() == self.scene:
                self.instruction_label.setPlainText(text)
                
                # Position at top-center of view
                scene_rect = self.scene.sceneRect()
                if not scene_rect.isEmpty():
                    self.instruction_label.setPos(
                        scene_rect.width()/2 - self.instruction_label.boundingRect().width()/2,
                        20
                    )
        except Exception as e:
            logging.error(f"Error updating instructions: {str(e)}")

    def wheelEvent(self, event):
        # Disable mouse wheel zoom
        event.ignore()

    def mousePressEvent(self, event):
        if self.waiting_for_reference and event.button() == Qt.LeftButton:
            # Handle reference point selection
            pos = self.mapToScene(event.pos())
            self.add_reference_circle(pos)
            self.waiting_for_reference = False
            self.update_instructions("Hold Shift and drag to draw ROIs\nDouble-click to complete ROI\nCtrl+click to delete ROI")
            return
            
        if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
            # Start new ROI with Shift + left click
            pos = self.mapToScene(event.pos())
            self.start_drawing(pos)
            event.accept()
        elif event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.ControlModifier:
                self.delete_roi_at_pos(self.mapToScene(event.pos()))
            else:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Complete ROI on double click"""
        if self.drawing and len(self.current_points) >= 3:
            self.complete_roi()

    def mouseMoveEvent(self, event):
        if self.drawing:
            pos = self.mapToScene(event.pos())
            self.update_drawing(pos)
            
            # Check for auto-completion
            try:
                if len(self.current_points) >= self.min_points_for_completion:
                    if self.near_start(pos):
                        self.complete_roi()
            except Exception as e:
                logging.error(f"Error in auto-completion: {str(e)}")
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def start_drawing(self, pos):
        self.drawing = True
        self.start_point = pos
        self.current_points = [pos]
        
        # Create new path with zoom-adjusted pen
        transform = self.transform()
        scale = transform.m11()
        
        pen = QPen(self.drawing_pen)
        pen.setWidth(max(1, int(1 / scale)))
        pen.setCosmetic(True)
        
        path = QPainterPath()
        path.moveTo(pos)
        self.current_path = self.scene.addPath(path, pen)
        
        # Add start point marker with scaled size
        scaled_radius = self.completion_radius / scale
        self.completion_circle = self.scene.addEllipse(
            pos.x() - scaled_radius,
            pos.y() - scaled_radius,
            scaled_radius * 2,
            scaled_radius * 2,
            self.completion_pen
        )

    def update_drawing(self, pos):
        if not self.drawing:
            return
        
        # Add point to list
        self.current_points.append(pos)
        
        # Update path with view transform compensation
        path = QPainterPath()
        path.moveTo(self.current_points[0])
        
        # Get current view transform
        transform = self.transform()
        scale = transform.m11()  # Get horizontal scale factor
        
        # Adjust line width based on zoom level
        pen = QPen(self.drawing_pen)
        pen.setWidth(max(1, int(1 / scale)))
        pen.setCosmetic(True)
        
        # Draw path
        for point in self.current_points[1:]:
            path.lineTo(point)
        
        # Remove old path and create new one
        if self.current_path:
            self.scene.removeItem(self.current_path)
        self.current_path = self.scene.addPath(path, pen)
        
        # Update completion circle size based on zoom
        if hasattr(self, 'completion_circle'):
            self.scene.removeItem(self.completion_circle)
            scaled_radius = self.completion_radius / scale
            self.completion_circle = self.scene.addEllipse(
                self.start_point.x() - scaled_radius,
                self.start_point.y() - scaled_radius,
                scaled_radius * 2,
                scaled_radius * 2,
                self.completion_pen
            )

    def near_start(self, pos):
        """Check if current position is near the starting point"""
        if not self.start_point:
            return False
        distance = ((pos.x() - self.start_point.x())**2 + 
                   (pos.y() - self.start_point.y())**2)**0.5
        return distance < self.completion_threshold

    def smooth_polygon(self, points):
        """Smooth polygon using spline interpolation with enhanced smoothing"""
        try:
            if len(points) < 4:
                return points
            
            # Convert points to numpy arrays
            x = np.array([p.x() for p in points])
            y = np.array([p.y() for p in points])
            
            # Ensure the polygon is closed
            x = np.append(x, [x[0], x[1]])
            y = np.append(y, [y[0], y[1]])
            
            # Create parameter array
            t = np.linspace(0, 1, len(x))
            
            try:
                # Create periodic spline with enhanced smoothing
                tck, u = splprep([x, y], u=t, s=self.smoothing_factor, per=True, k=3)
                
                # Generate more points for smoother curves
                u_new = np.linspace(0, 1, self.interpolation_points)
                smooth_x, smooth_y = splev(u_new, tck)
                
                # Create smooth points
                smooth_points = [QPointF(x, y) for x, y in zip(smooth_x, smooth_y)]
                
                # Ensure closure
                smooth_points.append(smooth_points[0])
                
                return smooth_points
                
            except ValueError:
                return points
                
        except Exception as e:
            logging.error(f"Error in smooth_polygon: {str(e)}")
            return points

    def complete_roi(self):
        """Complete current ROI drawing"""
        if not self.drawing or len(self.current_points) < 3:
            return
        
        try:
            # Create smooth path with curved segments
            smooth_points = self.create_smooth_roi()
            if smooth_points is None:
                return
            
            # Create path for final ROI
            path = QPainterPath()
            if smooth_points:
                path.moveTo(smooth_points[0])
                
                # Use curves instead of straight lines
                for i in range(1, len(smooth_points)-2, 2):
                    path.quadTo(
                        smooth_points[i].x(), smooth_points[i].y(),
                        smooth_points[i+1].x(), smooth_points[i+1].y()
                    )
                
                # Close the path smoothly
                path.quadTo(
                    smooth_points[-2].x(), smooth_points[-2].y(),
                    smooth_points[0].x(), smooth_points[0].y()
                )
            
            # Get color from region_colors instead of group_colors
            color = QColor(self.region_colors[self.current_group])
            
            # Create ROI with matching color and thicker line
            pen = QPen(color, 6)  # Increased from 4 to 6
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.RoundCap)  # Round line endings
            pen.setJoinStyle(Qt.RoundJoin)  # Round corners
            roi_item = self.scene.addPath(path, pen)
            
            # Create ROI number label
            roi_num = len(self.rois) + 1
            label_text = str(roi_num)
            label_item = self.scene.addText(label_text)
            
            # Style and position label with matching color
            font = label_item.font()
            font.setPointSize(32)
            font.setBold(True)
            label_item.setFont(font)
            label_item.setDefaultTextColor(color)
            
            # Position label
            rightmost_x = max(p.x() for p in smooth_points)
            rightmost_points = [p for p in smooth_points if p.x() == rightmost_x]
            rightmost_y = sum(p.y() for p in rightmost_points) / len(rightmost_points)
            label_item.setPos(rightmost_x + 20, rightmost_y - label_item.boundingRect().height()/2)
            
            # Make label always visible
            label_item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            
            # Store ROI
            self.rois.append({
                'item': roi_item,
                'points': smooth_points,
                'group': self.current_group,
                'label': label_item
            })
            
            # Cleanup
            if self.current_path:
                self.scene.removeItem(self.current_path)
                self.current_path = None
            if hasattr(self, 'completion_circle'):
                self.scene.removeItem(self.completion_circle)
            
            # Reset drawing state
            self.drawing = False
            self.current_points = []
            
            logging.info(f"Completed ROI {roi_num} in group {self.current_group}")
            
        except Exception as e:
            logging.error(f"Error in complete_roi: {str(e)}")
            self.drawing = False
            self.current_points = []
            if self.current_path:
                self.scene.removeItem(self.current_path)
                self.current_path = None

    def create_smooth_roi(self):
        """Create smoothed version of ROI points"""
        try:
            if len(self.current_points) < 3:
                return None
            
            # Convert QPointF to numpy arrays
            points = np.array([(p.x(), p.y()) for p in self.current_points])
            
            # Close the curve if not already closed
            if not np.allclose(points[0], points[-1]):
                points = np.vstack([points, points[0]])
            
            # Separate x and y coordinates
            x = points[:, 0]
            y = points[:, 1]
            
            # Create parameterization
            t = np.zeros(len(points))
            for i in range(1, len(points)):
                t[i] = t[i-1] + np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
            t = t/t[-1]
            
            # Fit splines with error handling
            try:
                tck, u = splprep([x, y], u=t, s=self.smoothing_factor, per=1)
                # Generate more points for smooth appearance
                u_new = np.linspace(0, 1.0, self.interpolation_points)
                smooth_x, smooth_y = splev(u_new, tck)
            except Exception as e:
                logging.warning(f"Spline fitting failed, using original points: {str(e)}")
                # Fall back to original points if smoothing fails
                smooth_x, smooth_y = x, y
            
            # Convert back to QPointF
            return [QPointF(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]
            
        except Exception as e:
            logging.error(f"Error in create_smooth_roi: {str(e)}")
            return None

    def cancel_drawing(self):
        self.cleanup_drawing()

    def cleanup_drawing(self):
        if self.current_path:
            self.scene.removeItem(self.current_path)
        if hasattr(self, 'completion_circle'):
            self.scene.removeItem(self.completion_circle)
        self.drawing = False
        self.current_path = None
        self.start_point = None
        self.current_points = []

    def delete_roi_at_pos(self, pos):
        """Delete ROI and its label at the given position"""
        for i, roi in enumerate(self.rois):
            if roi['item'].contains(pos):
                # Remove polygon
                self.scene.removeItem(roi['item'])
                # Remove label
                self.scene.removeItem(roi['label'])
                # Remove from list
                self.rois.pop(i)
                # Renumber remaining ROIs
                self.renumber_rois()
                logging.info(f"Deleted ROI {i+1}")
                break

    def renumber_rois(self):
        """Update ROI numbers after deletion"""
        for i, roi in enumerate(self.rois, 1):
            # Update label text
            roi['label'].setPlainText(str(i))
            # Update label color to match ROI
            color = QColor.fromHsv(i * 30 % 360, 255, 255)
            roi['label'].setDefaultTextColor(color)
            # Update ROI color
            pen = QPen(color, 4)
            pen.setCosmetic(True)
            roi['item'].setPen(pen)

    def get_mask(self):
        """Convert ROIs to binary mask"""
        if not self.rois:
            return None
            
        # Get image dimensions from scene rect
        rect = self.scene.sceneRect()
        h, w = int(rect.height()), int(rect.width())
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add each ROI to mask with unique ID
        for i, roi in enumerate(self.rois, 1):
            points = roi['points']
            # Convert points to numpy arrays
            x = [p.x() for p in points]
            y = [p.y() for p in points]
            rr, cc = polygon(y, x)
            # Clip to image bounds
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            mask[rr[valid], cc[valid]] = i
            
        return mask

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.scale(1.2, 1.2)
        elif event.key() == Qt.Key_Minus:
            self.scale(1/1.2, 1/1.2)
        elif event.key() == Qt.Key_Delete:
            # Delete selected ROI if any
            for roi in self.rois:
                if roi['item'].isSelected():
                    self.scene.removeItem(roi['item'])
                    self.scene.removeItem(roi['label'])  # Remove label
                    self.rois.remove(roi)
                    break
        super().keyPressEvent(event)

    def add_reference_circle(self, center):
        """Add reference circles with 800 and 1600 micrometer radii"""
        try:
            # Clear old circles
            self.clear_reference_circles()
            
            # Set up pen for dashed orange lines with thicker width
            pen = QPen(QColor(255, 165, 0))  # Orange color
            pen.setStyle(Qt.DashLine)
            pen.setWidth(6)  # Increased from 2 to 6
            pen.setCosmetic(True)
            
            # Calculate pixel radius for 800 micrometers
            base_radius_um = 800
            pixel_size = float(self.nd2_handler.pixel_size_um)
            base_radius_pixels = base_radius_um / pixel_size
            
            # Add two circles with increasing radii
            for multiplier in [1, 2]:
                radius = base_radius_pixels * multiplier
                circle = self.scene.addEllipse(
                    center.x() - radius,
                    center.y() - radius,
                    radius * 2,
                    radius * 2,
                    pen
                )
                self.reference_circles.append(circle)
                
                # Add radius label
                radius_um = base_radius_um * multiplier
                label = self.scene.addText(f"{radius_um}µm")
                label.setDefaultTextColor(QColor(255, 165, 0))
                label.setPos(
                    center.x() - label.boundingRect().width()/2,
                    center.y() + radius + 5
                )
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                self.reference_circles.append(label)
            
            logging.info("Added reference circles (800, 1600 µm)")
            
            # Update instructions immediately instead of using timer
            self.update_instructions(
                "Hold Shift to draw ROIs (auto-completes near start point)\n"
                "Ctrl+click to delete ROI\n"
                "+/- keys to zoom"
            )
            
        except Exception as e:
            logging.error(f"Error adding reference circles: {str(e)}")

    def clear_reference_circles(self):
        """Clear all reference circles and labels"""
        try:
            for item in self.reference_circles:
                if item is not None:
                    self.scene.removeItem(item)
            self.reference_circles.clear()
        except Exception as e:
            logging.error(f"Error clearing reference circles: {str(e)}")

    def prepare_image_for_display(self, image, channel_type='cell'):
        """Prepare image for display with black and white"""
        img = image.astype(float)
        img = (img - img.min()) / (img.max() - img.min())
        img = 1 - img  # Invert image
        
        # Convert to grayscale (black and white)
        rgb = np.stack([img, img, img], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        
        return rgb

class MainWindow(QMainWindow):
    # Define class-level attributes for regions and colors
    regions = [
        'Temporal_Peripheral', 'Temporal_Middle', 'Temporal_Central',
        'Superior_Peripheral', 'Superior_Middle', 'Superior_Central', 
        'Inferior_Peripheral', 'Inferior_Middle', 'Inferior_Central',
        'Nasal_Peripheral', 'Nasal_Middle', 'Nasal_Central'
    ]
    
    region_colors = {
        'Temporal_Peripheral': '#E76F51',  # Terracotta
        'Temporal_Middle': '#F4A261',      # Sandy brown
        'Temporal_Central': '#E67E22',     # Darker coral
        
        'Superior_Peripheral': '#1D3557',  # Dark blue
        'Superior_Middle': '#457B9D',      # Steel blue
        'Superior_Central': '#2E86C1',     # Darker blue
        
        'Inferior_Peripheral': '#2D6A4F',  # Forest green
        'Inferior_Middle': '#40916C',      # Medium green
        'Inferior_Central': '#2E7D32',     # Darker green
        
        'Nasal_Peripheral': '#7B2CBF',     # Deep purple
        'Nasal_Middle': '#9D4EDD',         # Medium purple
        'Nasal_Central': '#8E44AD'         # Darker purple
    }

    def __init__(self, nd2_path, batch_mode=False):
        super().__init__()
        self.nd2_path = Path(nd2_path)
        self.nd2_handler = None
        self.red_channel = None
        self.dapi_channel = None
        self.batch_mode = batch_mode
        self.current_group = "Temporal_Peripheral"  # Default group
        # Create color map for groups
        self.group_colors = plt.cm.tab20(np.linspace(0, 1, len(self.regions)))
        self.initUI()
        self.load_image()

    def initUI(self):
        self.setWindowTitle('ND2 ROI Selector')
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create horizontal layout for main content
        main_layout = QHBoxLayout(central_widget)
        
        # Create vertical layout for group buttons
        group_layout = QVBoxLayout()
        
        # Add group label
        group_label = QLabel("Select Group:")
        group_label.setStyleSheet("font-weight: bold;")
        group_layout.addWidget(group_label)
        
        # Add group buttons with anatomical names
        self.group_buttons = []
        for region in self.regions:
            btn = QPushButton(region.replace('_', ' '))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, x=region: self.set_group(x))
            if region == self.current_group:
                btn.setChecked(True)
            
            # Get color for this region
            color = self.region_colors[region]
            
            # Create button style without comments
            btn.setStyleSheet(f"""
                QPushButton {{
                    min-width: 150px;
                    padding: 5px;
                    margin: 2px;
                    background-color: {color};
                    color: white;
                }}
                QPushButton:checked {{
                    border: 3px solid black;
                }}
            """)
            group_layout.addWidget(btn)
            self.group_buttons.append(btn)
        
        # Add stretch to push buttons to top
        group_layout.addStretch()
        
        # Create vertical layout for viewer and toolbar
        viewer_layout = QVBoxLayout()
        
        # Modify toolbar
        toolbar = QHBoxLayout()
        
        clear_btn = QPushButton('Clear ROIs')
        clear_btn.clicked.connect(self.clear_rois)
        toolbar.addWidget(clear_btn)
        
        confirm_btn = QPushButton('Confirm Selection')
        confirm_btn.clicked.connect(self.confirm_selection)
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        toolbar.addWidget(confirm_btn)
        
        viewer_layout.addLayout(toolbar)
        
        # Create image viewer
        self.viewer = ImageViewer()
        self.viewer.current_group = self.current_group  # Pass initial group
        viewer_layout.addWidget(self.viewer)
        
        # Add instruction panel
        instruction_text = (
            "Instructions:\n"
            "1. Click center of retina for reference circles\n"
            "2. Select a group from the left panel\n"
            "3. Hold Shift+left click to draw ROIs (Double click or auto-complete near start point)\n"
            "4. Ctrl+click to delete ROI\n"
            "5. +/- keys to zoom\n"
            "6. Drag to pan"
        )
        instructions = QLabel(instruction_text)
        instructions.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        viewer_layout.addWidget(instructions)
        
        # Add layouts to main layout
        main_layout.addLayout(group_layout)
        main_layout.addLayout(viewer_layout, stretch=1)
        
        self.setGeometry(100, 100, 1400, 800)

    def set_group(self, group_name):
        """Set current group and update button states"""
        self.current_group = group_name
        self.viewer.current_group = group_name
        
        # Update button states
        for i, btn in enumerate(self.group_buttons):
            btn.setChecked(btn.text().replace(' ', '_') == group_name)

    def clear_rois(self):
        for roi in self.viewer.rois:
            self.viewer.scene.removeItem(roi['item'])  # Remove polygon
            self.viewer.scene.removeItem(roi['label'])  # Remove label
        self.viewer.rois.clear()
        logging.info("Cleared all ROIs")

    def load_image(self):
        try:
            # Load ND2 file
            self.nd2_handler = ND2Handler(str(self.nd2_path))
            self.red_channel, self.dapi_channel = self.nd2_handler.read_channels()
            
            # Pass nd2_handler to viewer
            self.viewer.nd2_handler = self.nd2_handler
            
            logging.info(f"Loaded image shape: {self.red_channel.shape}")
            logging.info(f"Value range: [{self.red_channel.min():.2f}, {self.red_channel.max():.2f}]")
            
            # Normalize and prepare for display
            image = self.red_channel.copy()
            p2, p98 = np.percentile(image, (2, 98))
            image = np.clip(image, p2, p98)
            image = ((image - p2) / (p98 - p2))  # Scale to [0,1]
            
            # Convert to RGB with white background using viewer's method
            rgb_image = self.viewer.prepare_image_for_display(image, 'cell')
            
            # Create QImage
            h, w = rgb_image.shape[:2]
            bytes_per_line = 3 * w
            qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Display image
            pixmap = QPixmap.fromImage(qimage)
            self.viewer.scene.clear()
            self.viewer.scene.addPixmap(pixmap)
            self.viewer.setSceneRect(QRectF(pixmap.rect()))
            self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)
            
            # Only show popup in non-batch mode
            if not self.batch_mode:
                QMessageBox.information(
                    self,
                    "Getting Started",
                    "1. First, click on the center of the retina to add reference circles\n\n"
                    "2. To draw ROIs:\n"
                    "   - Hold Shift and start drawing\n"
                    "   - Drawing will auto-complete when near start point\n\n"
                    "3. Navigation:\n"
                    "   - Use +/- keys to zoom\n"
                    "   - Drag to pan\n"
                    "   - Ctrl+click to delete ROI",
                    QMessageBox.Ok
                )
            
            # Update status bar
            self.statusBar().showMessage(
                'Click on the center of retina for reference circles, then use Shift+click to draw ROIs'
            )
            
            logging.info("Image displayed successfully")
            
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            raise

    def confirm_selection(self):
        """Save ROIs and close window in batch mode"""
        if not self.viewer.rois:
            QMessageBox.warning(self, "No ROIs", "Please select at least one ROI first.")
            return
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = self.nd2_path.parent / f"{self.nd2_path.stem}_rois_{timestamp}.json"
            
            # Prepare ROI data with group information
            roi_data = []
            for i, roi in enumerate(self.viewer.rois, 1):
                points = roi['points']
                polygon_points = [(int(p.x()), int(p.y())) for p in points]
                roi_data.append({
                    'roi_index': i,
                    'group': roi['group'],  # Include group information
                    'polygon': polygon_points
                })
            
            # Save ROIs with metadata
            metadata = {
                'source_file': str(self.nd2_path),
                'pixel_size_um': self.nd2_handler.pixel_size_um,
                'image_shape': self.red_channel.shape,
                'selection_time': timestamp,
                'groups': [f"group{i+1}" for i in range(12)],  # Add groups list
                'rois': roi_data
            }
            
            with open(save_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Save visualization with matching timestamp
            vis_path = self.nd2_path.parent / f"{self.nd2_path.stem}_rois_{timestamp}_visualization.png"
            self.save_visualization_to_path(vis_path)
            
            logging.info(f"Saved {len(roi_data)} ROIs to {save_path}")
            
            # In batch mode, just close the window
            if self.batch_mode:
                self.close()
            else:
                # Ask to continue with analysis only in non-batch mode
                reply = QMessageBox.question(
                    self,
                    'Continue to Analysis',
                    'ROIs saved successfully. Would you like to run the analysis now?',
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.close()
                    if hasattr(self, 'parent_tool'):
                        self.parent_tool.run_analysis_with_roi(str(save_path))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving ROIs: {str(e)}")

    def save_visualization_to_path(self, save_path):
        """Save thumbnail of image with ROIs to specified path"""
        if not self.viewer.scene.items():
            logging.warning("No image to save")
            return
        
        try:
            # Create a 3000px width thumbnail
            target_width = 3000
            scene_rect = self.viewer.scene.sceneRect()
            aspect_ratio = scene_rect.height() / scene_rect.width()
            target_height = int(target_width * aspect_ratio)
            
            # Create pixmap for thumbnail
            pixmap = QPixmap(target_width, target_height)
            pixmap.fill(Qt.black)
            
            # Create painter
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Calculate scale to fit scene in thumbnail
            scale = target_width / scene_rect.width()
            painter.scale(scale, scale)
            
            # Render scene
            self.viewer.scene.render(
                painter,
                QRectF(0, 0, scene_rect.width(), scene_rect.height()),  # target
                scene_rect  # source
            )
            painter.end()
            
            # Save the thumbnail
            pixmap.save(str(save_path))
            
        except Exception as e:
            logging.error(f"Error saving visualization: {str(e)}")

    def closeEvent(self, event):
        """Handle window closing"""
        event.accept()  # Always accept closing, no need to ask about saving

def main():
    app = QApplication(sys.argv)
    
    if len(sys.argv) != 2:
        print("Usage: python interactive_roi_selector.py path/to/file.nd2")
        sys.exit(1)
    
    window = MainWindow(sys.argv[1])
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 