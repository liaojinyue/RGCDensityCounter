import numpy as np
from skimage import filters, measure
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from cellpose import models
from nd2handler import ND2Handler  # Import the new handler
from skimage.segmentation import find_boundaries
from multiprocessing import Pool
import math
from tqdm import tqdm
from time import time
import humanize
from datetime import datetime
import logging
import time
import psutil
import gc
import torch
import json
from matplotlib.backends.backend_pdf import PdfPages
from skimage.morphology import dilation, square
from skimage.draw import polygon

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")

class RetinalImageAnalyzer:
    def __init__(self, input_path, test_mode=False):
        self.input_path = Path(input_path)
        self.nd2_handler = ND2Handler(str(self.input_path))
        self.test_mode = test_mode
        self.pixel_size = None
        self.red_channel = None
        self.dapi_channel = None
        
        # Initialize Cellpose models
        try:
            # Try to load custom RGC model
            self.cell_model = models.CellposeModel(pretrained_model='./models/rgc/rgc.zip', gpu=False)
            logging.info("Loaded custom RGC model")
        except Exception as e:
            # Fall back to cyto2 if custom model not found
            logging.warning(f"Could not load custom RGC model: {str(e)}")
            logging.info("Falling back to cyto2 model")
            self.cell_model = models.Cellpose(model_type='cyto2', gpu=False)

        # For nuclei detection (if needed)
        self.nuclei_model = models.Cellpose(model_type='nuclei', gpu=False)

        # Updated parameters
        self.cell_params = {
            'diameter': 30,
            'flow_threshold': 0.8,
            'cellprob_threshold': 1.0
        }
        self.nuclei_params = {
            'diameter': 20,
            'flow_threshold': 0.8,
            'cellprob_threshold': -0.5
        }

    def load_image(self):
        """Load image data from nd2 file"""
        # Get pixel size
        self.pixel_size = self.nd2_handler.pixel_size_um
        logging.info(f"Pixel size: {self.pixel_size} µm")
        
        # Load channels
        self.red_channel, self.dapi_channel = self.nd2_handler.read_channels()
        
        if self.test_mode:
            # Crop center region for testing
            h, w = self.red_channel.shape
            y1 = h//2 - 256
            x1 = w//2 - 256
            self.red_channel = self.red_channel[y1:y1+512, x1:x1+512]
            self.dapi_channel = self.dapi_channel[y1:y1+512, x1:x1+512]
        
        # Log channel information
        logging.info(f"Red channel shape: {self.red_channel.shape}")
        logging.info(f"DAPI channel shape: {self.dapi_channel.shape}")

    def analyze_roi(self, roi_index, roi_mask, red_channel, dapi_channel):
        """Analyze a single ROI"""
        # Extract ROI region
        roi_region = roi_mask == roi_index
        
        # Get bounding box for efficiency
        props = measure.regionprops(roi_region.astype(int))[0]
        bbox = props.bbox
        
        # Extract and copy ROI regions
        self.red_channel = red_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
        self.dapi_channel = dapi_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
        
        # Apply ROI mask
        roi_mask_cropped = roi_region[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        self.red_channel *= roi_mask_cropped
        self.dapi_channel *= roi_mask_cropped
        
        # Check for empty ROIs
        if np.sum(self.red_channel) == 0 or np.sum(self.dapi_channel) == 0:
            logging.warning(f"Empty ROI detected: {roi_index}")
            return None
        
        # Run detection
        try:
            cell_masks, nuclei_masks = self.detect_cells_and_nuclei()
            metrics = self.calculate_metrics(cell_masks, nuclei_masks)
            overlap_summary, overlap_details = self.analyze_cell_nuclei_overlap(cell_masks, nuclei_masks)
            
            return {
                'roi_index': roi_index,
                'bbox': bbox,
                'metrics': metrics,
                'overlap_summary': overlap_summary,
                'overlap_details': overlap_details
            }
        except Exception as e:
            logging.error(f"Error analyzing ROI {roi_index}: {str(e)}")
            return None

    def detect_cells_and_nuclei(self):
        """Detect both cells and nuclei"""
        if self.red_channel is None or self.dapi_channel is None:
            raise ValueError("Image data not loaded. Call load_image() first.")
        
        try:
            # Ensure proper shape for Cellpose
            red = self.red_channel.astype(np.float32)
            dapi = self.dapi_channel.astype(np.float32)
            
            # Add channel dimension if needed
            if red.ndim == 2:
                red = red[..., np.newaxis]
            if dapi.ndim == 2:
                dapi = dapi[..., np.newaxis]
            
            # Detect cells (red channel)
            cell_output = self.cell_model.eval(
                [red],
                diameter=self.cell_params['diameter'],
                flow_threshold=self.cell_params['flow_threshold'],
                cellprob_threshold=self.cell_params['cellprob_threshold'],
                channels=[0,0]
            )
            
            # Detect nuclei (DAPI channel)
            nuclei_output = self.nuclei_model.eval(
                [dapi],
                diameter=self.nuclei_params['diameter'],
                flow_threshold=self.nuclei_params['flow_threshold'],
                cellprob_threshold=self.nuclei_params['cellprob_threshold'],
                channels=[0,0]
            )
            
            # Process outputs
            cell_masks = cell_output[0] if isinstance(cell_output, tuple) else cell_output
            nuclei_masks = nuclei_output[0] if isinstance(nuclei_output, tuple) else nuclei_output
            
            # Ensure we have numpy arrays
            cell_masks = np.array(cell_masks[0] if isinstance(cell_masks, list) else cell_masks, dtype=np.int32)
            nuclei_masks = np.array(nuclei_masks[0] if isinstance(nuclei_masks, list) else nuclei_masks, dtype=np.int32)
            
            logging.info(f"Detected {len(np.unique(cell_masks)) - 1} cells")
            logging.info(f"Detected {len(np.unique(nuclei_masks)) - 1} nuclei")
            
            return cell_masks, nuclei_masks
            
        except Exception as e:
            logging.error(f"Error in cell detection: {str(e)}")
            logging.error("Traceback:", exc_info=True)
            raise

    def calculate_metrics(self, cell_masks, nuclei_masks):
        """Calculate density and overlap metrics"""
        if self.pixel_size is None:
            raise ValueError("Pixel size not available. Call load_image() first.")
            
        cell_props = measure.regionprops(cell_masks)
        nuclei_props = measure.regionprops(nuclei_masks)
        
        # Calculate area in mm²
        pixel_area = self.red_channel.shape[0] * self.red_channel.shape[1]
        area_mm2 = pixel_area * (self.pixel_size/1000)**2
        
        # Count cells with nuclei
        cells_with_nuclei = 0
        for cell_prop in cell_props:
            cell_mask = cell_masks == cell_prop.label
            for nuclei_prop in nuclei_props:
                nuclei_mask = nuclei_masks == nuclei_prop.label
                if np.any(cell_mask & nuclei_mask):
                    cells_with_nuclei += 1
                    break
        
        return {
            'total_cells': len(cell_props),
            'total_nuclei': len(nuclei_props),
            'cells_with_nuclei': cells_with_nuclei,
            'area_mm2': area_mm2,
            'cell_density_mm2': len(cell_props) / area_mm2,
            'nuclei_density_mm2': len(nuclei_props) / area_mm2,
            'validated_cell_density_mm2': cells_with_nuclei / area_mm2
        }

    def analyze_cell_nuclei_overlap(self, cell_masks, nuclei_masks):
        """
        Analyze detailed overlap between cells and nuclei
        Returns detailed metrics about the overlap relationships
        """
        cell_props = measure.regionprops(cell_masks)
        nuclei_props = measure.regionprops(nuclei_masks)
        
        overlap_data = []
        cells_with_nuclei = 0
        cells_without_nuclei = 0
        cells_with_multiple_nuclei = 0
        
        for cell_prop in cell_props:
            cell_mask = cell_masks == cell_prop.label
            cell_area = np.sum(cell_mask)
            nuclei_overlaps = []
            
            for nuclei_prop in nuclei_props:
                nuclei_mask = nuclei_masks == nuclei_prop.label
                overlap_area = np.sum(cell_mask & nuclei_mask)
                
                if overlap_area > 0:
                    overlap_ratio = overlap_area / min(cell_area, np.sum(nuclei_mask))
                    nuclei_overlaps.append({
                        'nuclei_id': nuclei_prop.label,
                        'overlap_area': overlap_area,
                        'overlap_ratio': overlap_ratio
                    })
            
            cell_data = {
                'cell_id': cell_prop.label,
                'cell_area': cell_area,
                'centroid_y': cell_prop.centroid[0],
                'centroid_x': cell_prop.centroid[1],
                'n_nuclei': len(nuclei_overlaps),
                'nuclei_overlaps': nuclei_overlaps
            }
            overlap_data.append(cell_data)
            
            # Update counts
            if len(nuclei_overlaps) == 0:
                cells_without_nuclei += 1
            elif len(nuclei_overlaps) == 1:
                cells_with_nuclei += 1
            else:
                cells_with_multiple_nuclei += 1
        
        # Calculate summary metrics
        total_cells = len(cell_props)
        total_nuclei = len(nuclei_props)
        
        summary = {
            'total_cells': total_cells,
            'total_nuclei': total_nuclei,
            'cells_with_nuclei': cells_with_nuclei,
            'cells_without_nuclei': cells_without_nuclei,
            'cells_with_multiple_nuclei': cells_with_multiple_nuclei,
            'percent_cells_with_nuclei': (cells_with_nuclei / total_cells * 100) if total_cells > 0 else 0,
            'percent_cells_with_multiple': (cells_with_multiple_nuclei / total_cells * 100) if total_cells > 0 else 0
        }
        
        return summary, overlap_data

    def prepare_image_for_display(self, image, channel_type='cell'):
        """Prepare image for display with white background"""
        img = image.astype(float)
        img = (img - img.min()) / (img.max() - img.min())
        img = 1 - img  # Invert image
        
        rgb = np.ones((*img.shape, 3))
        
        if channel_type == 'nuclei':
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = 1.0
        else:
            rgb[:, :, 0] = 1.0
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
        
        return rgb

    def generate_analysis_report(self, save_dir=None):
        """Generate comprehensive analysis report with visualizations"""
        if save_dir is None:
            save_dir = Path('analysis_results')
        save_dir.mkdir(exist_ok=True)
        
        # Detect cells and nuclei
        cell_masks, nuclei_masks = self.detect_cells_and_nuclei()
        
        # Calculate metrics
        metrics = self.calculate_metrics(cell_masks, nuclei_masks)
        overlap_summary, overlap_details = self.analyze_cell_nuclei_overlap(cell_masks, nuclei_masks)
        
        # Get cell and nuclei properties
        cell_props = measure.regionprops(cell_masks)
        nuclei_props = measure.regionprops(nuclei_masks)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Inverted cell image (red on white)
        cell_rgb = self.prepare_image_for_display(self.red_channel, 'cell')
        axes[0,0].imshow(cell_rgb)
        axes[0,0].set_title('Cell Channel')
        axes[0,0].axis('off')
        
        # 2. Inverted nuclei image (blue on white)
        nuclei_rgb = self.prepare_image_for_display(self.dapi_channel, 'nuclei')
        axes[0,1].imshow(nuclei_rgb)
        axes[0,1].set_title('Nuclei Channel')
        axes[0,1].axis('off')
        
        # 3. Cell segmentation with outlines and centroids
        axes[0,2].imshow(cell_rgb)
        boundaries = find_boundaries(cell_masks)
        axes[0,2].imshow(boundaries, alpha=0.5, cmap='Reds')
        for prop in cell_props:
            y, x = prop.centroid
            axes[0,2].plot(x, y, '+', color='blue', markersize=5, markeredgewidth=2)
        axes[0,2].set_title(f'Cell Segmentation (n={len(cell_props)})')
        axes[0,2].axis('off')
        
        # 4. Nuclei segmentation with outlines and centroids
        axes[1,0].imshow(nuclei_rgb)
        boundaries = find_boundaries(nuclei_masks)
        axes[1,0].imshow(boundaries, alpha=0.5, cmap='Blues')
        for prop in nuclei_props:
            y, x = prop.centroid
            axes[1,0].plot(x, y, '+', color='blue', markersize=10, markeredgewidth=2)
        axes[1,0].set_title(f'Nuclei Segmentation (n={len(nuclei_props)})')
        axes[1,0].axis('off')
        
        # 5. Overlapping segmentation outlines
        axes[1,1].imshow(np.ones_like(cell_rgb))  # White background
        cell_boundaries = find_boundaries(cell_masks)
        nuclei_boundaries = find_boundaries(nuclei_masks)
        axes[1,1].imshow(cell_boundaries, cmap='Reds', alpha=0.5)
        axes[1,1].imshow(nuclei_boundaries, cmap='Blues', alpha=0.5)
        axes[1,1].set_title('Overlapping Segmentation')
        axes[1,1].axis('off')
        
        # Add metrics summary
        metrics_text = '\n'.join([
            f"Total Cells: {metrics['total_cells']}",
            f"Total Nuclei: {metrics['total_nuclei']}",
            f"Cells with Nuclei: {metrics['cells_with_nuclei']}",
            f"Cell Density: {metrics['cell_density_mm2']:.1f} cells/mm²",
            f"Validated Cell Density: {metrics['validated_cell_density_mm2']:.1f} cells/mm²",
            f"Area: {metrics['area_mm2']:.2f} mm²"
        ])
        axes[1,2].text(0.1, 0.5, metrics_text, transform=axes[1,2].transAxes)
        axes[1,2].axis('off')
        
        plt.tight_layout()
        
        # Create PDF
        with PdfPages(save_dir / 'analysis_report.pdf') as pdf:
            # First page: Detailed statistics
            fig = plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            
            # Overall statistics
            stats_text = [
                "OVERALL STATISTICS",
                f"Total ROIs: {len(self.rois)}",
                f"Total Cells: {total_cells}",
                f"Average Density: {avg_density:.1f} cells/mm²",
                "\nPER ROI STATISTICS:"
            ]
            
            # Add per-ROI statistics
            for roi in self.rois:
                roi_stats = [
                    f"\nROI {roi['index']}:",
                    f"Group: {roi['group']}",
                    f"Cell Count: {roi['metrics']['total_cells']}",
                    f"Area: {roi['metrics']['area_mm2']:.2f} mm²",
                    f"Density: {roi['metrics']['cell_density_mm2']:.1f} cells/mm²"
                ]
                stats_text.extend(roi_stats)
            
            plt.text(0.1, 0.95, '\n'.join(stats_text), 
                    transform=fig.transFigure, 
                    fontsize=12,
                    verticalalignment='top')
            
            pdf.savefig(fig)
            plt.close()
            
            # Individual ROI pages
            for roi in self.rois:
                fig = self.generate_roi_visualization(roi)
                
                # Add ROI identification text
                plt.figtext(0.02, 0.98, 
                           f"ROI {roi['index']} - {roi['group']}", 
                           fontsize=14, 
                           weight='bold')
                
                pdf.savefig(fig)
                plt.close()

    def analyze_roi_with_mask(self, roi_index, roi_mask, results_dir):
        """Analyze a single ROI"""
        try:
            logging.info(f"\n=== Processing ROI {roi_index} ===")
            
            # Ensure mask and image have same shape
            if roi_mask.shape != self.red_channel.shape:
                logging.error(f"Mask shape {roi_mask.shape} doesn't match image shape {self.red_channel.shape}")
                return None
            
            # Get bounding box for efficiency
            props = measure.regionprops(roi_mask.astype(int))
            if not props:
                logging.warning(f"Empty ROI mask for ROI {roi_index}")
                return None
            
            bbox = props[0].bbox
            logging.info(f"ROI {roi_index} bbox: {bbox}")
            logging.info(f"ROI {roi_index} size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} pixels")
            
            # Create cropped versions of channels and mask
            roi_red = self.red_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
            roi_dapi = self.dapi_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()  # Add DAPI crop
            roi_mask_cropped = roi_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            # Apply mask to cropped regions
            roi_red[~roi_mask_cropped] = 0
            roi_dapi[~roi_mask_cropped] = 0  # Apply mask to DAPI
            
            # Store original channels
            original_red = self.red_channel
            original_dapi = self.dapi_channel
            
            try:
                # Set cropped regions for analysis
                self.red_channel = roi_red
                self.dapi_channel = roi_dapi  # Set DAPI channel
                
                # Run cell detection only
                cell_masks = self.detect_cells()
                
                # Calculate basic metrics
                cell_props = measure.regionprops(cell_masks)
                total_cells = len(cell_props)
                area_mm2 = (roi_red.shape[0] * roi_red.shape[1] * (self.pixel_size ** 2)) / 1e6
                cell_density = total_cells / area_mm2 if area_mm2 > 0 else 0
                
                metrics = {
                    'total_cells': total_cells,
                    'area_mm2': area_mm2,
                    'cell_density_mm2': cell_density
                }
                
                # Generate visualization
                fig = self.generate_visualization(cell_masks, metrics)
                
                # Save visualization with permanent name
                roi_vis_path = results_dir / f"roi_{roi_index}_analysis.png"
                fig.savefig(roi_vis_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logging.info(f"Saved ROI {roi_index} visualization to {roi_vis_path}")
                
                return {
                    'roi_index': roi_index,
                    'bbox': bbox,
                    'metrics': metrics,
                    'visualization': fig
                }
                
            finally:
                # Restore original channels
                self.red_channel = original_red
                self.dapi_channel = original_dapi
                
        except Exception as e:
            logging.error(f"Error analyzing ROI {roi_index}: {str(e)}")
            return None

    def detect_cells(self):
        """Detect cells only"""
        if self.red_channel is None:
            raise ValueError("Image data not loaded. Call load_image() first.")
        
        try:
            # Ensure proper shape for Cellpose
            red = self.red_channel.astype(np.float32)
            if red.ndim == 2:
                red = red[..., np.newaxis]
            
            # Detect cells
            cell_output = self.cell_model.eval(
                [red],
                diameter=self.cell_params['diameter'],
                flow_threshold=self.cell_params['flow_threshold'],
                cellprob_threshold=self.cell_params['cellprob_threshold'],
                channels=[0,0]
            )
            
            # Process output
            cell_masks = cell_output[0] if isinstance(cell_output, tuple) else cell_output
            cell_masks = np.array(cell_masks[0] if isinstance(cell_masks, list) else cell_masks, dtype=np.int32)
            
            logging.info(f"Detected {len(np.unique(cell_masks)) - 1} cells")
            
            return cell_masks
            
        except Exception as e:
            logging.error(f"Error in cell detection: {str(e)}")
            logging.error("Traceback:", exc_info=True)
            raise

    def generate_visualization(self, cell_masks, metrics):
        """Generate visualization using pre-computed masks and metrics"""
        # Calculate cell properties first
        cell_props = measure.regionprops(cell_masks)
        
        # Create figure with landscape orientation and high DPI
        fig = plt.figure(figsize=(11, 8.5), dpi=600)  # Increased DPI for better resolution
        
        # Create grid with optimized spacing
        gs = plt.GridSpec(2, 3, height_ratios=[0.85, 0.15])
        gs.update(wspace=0.1, hspace=0.1,
                 left=0.05, right=0.95,
                 top=0.95, bottom=0.05)
        
        # 1. Cell Channel
        ax1 = fig.add_subplot(gs[0, 0])
        cell_rgb = self.prepare_image_for_display(self.red_channel, 'cell')
        ax1.imshow(cell_rgb, interpolation='nearest')
        ax1.set_title('Cell Channel', pad=10, fontsize=14)
        ax1.axis('off')
        
        # 2. Cell segmentation with thicker boundaries
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cell_rgb, interpolation='nearest')
        
        # Create thicker boundaries
        boundaries = find_boundaries(cell_masks)
        thick_boundaries = dilation(boundaries, square(2))  # Make boundaries thicker
        
        # Create boundary overlay with thicker lines
        boundary_rgba = np.zeros((*thick_boundaries.shape, 4))
        boundary_rgba[thick_boundaries, 0] = 1.0  # Red channel
        boundary_rgba[thick_boundaries, 3] = 1.0  # Alpha channel
        
        ax2.imshow(boundary_rgba, interpolation='nearest')
        ax2.set_title(f'Cell Segmentation (n={len(cell_props)})', pad=10, fontsize=14)
        ax2.axis('off')
        
        # 3. Cell boundaries on DAPI
        ax3 = fig.add_subplot(gs[0, 2])
        dapi_rgb = self.prepare_image_for_display(self.dapi_channel, 'nuclei')
        ax3.imshow(dapi_rgb, interpolation='nearest')
        ax3.imshow(boundary_rgba, interpolation='nearest')
        ax3.set_title('Cell Boundaries on DAPI', pad=10, fontsize=14)
        ax3.axis('off')
        
        # 4. Metrics text
        ax4 = fig.add_subplot(gs[1, :])
        metrics_text = f"Total Cells: {metrics['total_cells']}    Cell Density: {metrics['cell_density_mm2']:.1f} cells/mm²    Area: {metrics['area_mm2']:.2f} mm²"
        ax4.text(0.5, 0.5, metrics_text,
                 transform=ax4.transAxes,
                 verticalalignment='center',
                 horizontalalignment='center',
                 fontsize=12)
        ax4.axis('off')
        
        # Ensure proper scaling
        for ax in [ax1, ax2, ax3]:
            ax.set_aspect('equal', adjustable='box')
        
        return fig

    def analyze_rois_by_group(self, roi_data, results_dir):
        """Analyze ROIs with group-based metrics"""
        # Extract unique groups
        groups = set(roi['group'] for roi in roi_data['rois'])
        
        # Initialize group metrics
        group_metrics = {group: {
            'total_cells': 0,
            'total_area': 0,
            'rois': [],
            'cell_densities': [],
            'roi_metrics': []
        } for group in groups}
        
        # Analyze each ROI and aggregate by group
        for roi in roi_data['rois']:
            roi_index = roi['roi_index']
            group = roi['group']
            polygon_points = roi['polygon']
            
            # Create and analyze ROI mask
            mask = np.zeros(self.red_channel.shape, dtype=bool)
            y_coords = [p[1] for p in polygon_points]
            x_coords = [p[0] for p in polygon_points]
            rr, cc = polygon(y_coords, x_coords)
            valid_points = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
            mask[rr[valid_points], cc[valid_points]] = True
            
            # Pass results_dir to analyze_roi_with_mask
            roi_results = self.analyze_roi_with_mask(roi_index, mask, results_dir)
            
            if roi_results is not None:
                metrics = roi_results['metrics']
                # Add results to group metrics
                group_metrics[group]['total_cells'] += metrics['total_cells']
                group_metrics[group]['total_area'] += metrics['area_mm2']
                group_metrics[group]['cell_densities'].append(metrics['cell_density_mm2'])
                group_metrics[group]['rois'].append(roi_index)
                group_metrics[group]['roi_metrics'].append({
                    'roi_index': roi_index,
                    **metrics
                })
        
        return group_metrics