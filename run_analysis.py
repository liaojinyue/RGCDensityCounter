import sys
from pathlib import Path
import numpy as np
from pipeline import RetinalImageAnalyzer
from roi_selector import ROISelector
import pandas as pd
import json
import matplotlib.pyplot as plt
import time
import logging
from skimage import measure
from skimage.segmentation import find_boundaries
from matplotlib.backends.backend_pdf import PdfPages
from skimage.draw import polygon

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.StreamHandler(),
                       logging.FileHandler('analysis_debug.log')
                   ])

def log_array_info(name, array):
    """Log array information for debugging"""
    logging.info(f"{name} - Shape: {array.shape}, Type: {array.dtype}, "
                f"Range: [{array.min():.2f}, {array.max():.2f}]")

def process_nd2_with_rois(nd2_path, roi_path=None, save_dir=None):
    """Process nd2 file with saved ROIs"""
    nd2_filename = Path(nd2_path).stem
    roi_filename = Path(roi_path).stem
    
    if save_dir is None:
        save_dir = Path('analysis_results')
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create results directory with filename and ROI info
    results_dir = save_dir / f"{nd2_filename}_{roi_filename}_results"
    results_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    logging.info("Initializing analyzer and loading image...")
    analyzer = RetinalImageAnalyzer(nd2_path)
    analyzer.load_image()
    
    # Load ROIs from JSON
    logging.info("Loading ROI data...")
    with open(roi_path) as f:
        roi_data = json.load(f)
    
    total_rois = len(roi_data['rois'])
    logging.info(f"Found {total_rois} ROIs to process")
    
    # Verify image shape matches
    if tuple(roi_data['image_shape']) != analyzer.red_channel.shape:
        logging.error(f"ROI image shape {roi_data['image_shape']} doesn't match loaded image {analyzer.red_channel.shape}")
        return
    
    # Prepare data for Excel
    all_metrics = []
    
    # Create PDF
    logging.info("Processing ROIs and generating report...")
    pdf_path = results_dir / f"{nd2_filename}_{roi_filename}_analysis_report.pdf"

    try:
        with PdfPages(pdf_path) as pdf:
            # Analyze ROIs by group
            group_results = analyzer.analyze_rois_by_group(roi_data, results_dir)
            
            # Save summary page
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Create summary text with group information
            summary_text = [f"Analysis Summary for {Path(nd2_path).name}\n\n"]
            
            # Add group summaries
            summary_text.append("Group Summaries:")
            for group, metrics in group_results.items():
                n_rois = len(metrics['rois'])
                if n_rois > 0:
                    avg_density = metrics['total_cells'] / metrics['total_area'] if metrics['total_area'] > 0 else 0
                    density_std = np.std(metrics['cell_densities']) if len(metrics['cell_densities']) > 1 else 0
                    
                    summary_text.append(f"\n{group}:")
                    summary_text.append(f"  ROIs analyzed: {n_rois}")
                    summary_text.append(f"  Total cells: {metrics['total_cells']}")
                    summary_text.append(f"  Total area: {metrics['total_area']:.2f} mm²")
                    summary_text.append(f"  Average density: {avg_density:.1f} ± {density_std:.1f} cells/mm²")
            
            # Add analysis parameters
            summary_text.append("\nAnalysis Parameters:")
            summary_text.append(f"Pixel size: {analyzer.pixel_size:.3f} µm")
            summary_text.append(f"Cell detection threshold: {analyzer.cell_params['cellprob_threshold']}")
            
            # Add text to figure
            ax.text(0.1, 0.95, '\n'.join(summary_text),
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    fontsize=10)
            
            # Save summary page
            pdf.savefig(fig, dpi=600, bbox_inches='tight', orientation='landscape')
            plt.close()
            
            # Add ROI visualizations
            for roi in roi_data['rois']:
                roi_index = roi['roi_index']
                roi_vis_path = results_dir / f"roi_{roi_index}_analysis.png"
                if roi_vis_path.exists():
                    img = plt.imread(str(roi_vis_path))
                    fig = plt.figure(figsize=(11, 8.5))
                    plt.imshow(img, interpolation='nearest')
                    plt.axis('off')
                    pdf.savefig(fig, 
                               dpi=600,
                               bbox_inches='tight',
                               orientation='landscape')
                    plt.close()
                    logging.info(f"Added ROI {roi_index} visualization to PDF")
                else:
                    logging.warning(f"ROI visualization not found: {roi_vis_path}")

    except Exception as e:
        logging.error(f"Error creating PDF: {str(e)}")
        if 'pdf' in locals():
            pdf.close()
        raise

    finally:
        plt.close('all')

    # Create Excel report with group information
    excel_path = results_dir / f"{Path(nd2_path).stem}_analysis_summary.xlsx"
    if group_results:
        with pd.ExcelWriter(excel_path) as writer:
            # Summary sheet
            summary_rows = []
            for group, metrics in group_results.items():
                for roi_metric in metrics['roi_metrics']:
                    summary_rows.append({
                        'Group': group,
                        'ROI': roi_metric['roi_index'],
                        'Total Cells': roi_metric['total_cells'],
                        'Cell Density (mm²)': roi_metric['cell_density_mm2'],
                        'Area (mm²)': roi_metric['area_mm2']
                    })
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='ROI Summary', index=False)
            
            # Group summary sheet
            group_summary_rows = []
            for group, metrics in group_results.items():
                n_rois = len(metrics['rois'])
                if n_rois > 0:
                    avg_density = metrics['total_cells'] / metrics['total_area'] if metrics['total_area'] > 0 else 0
                    density_std = np.std(metrics['cell_densities']) if len(metrics['cell_densities']) > 1 else 0
                    
                    group_summary_rows.append({
                        'Group': group,
                        'Number of ROIs': n_rois,
                        'Total Cells': metrics['total_cells'],
                        'Total Area (mm²)': metrics['total_area'],
                        'Average Density (cells/mm²)': avg_density,
                        'Density Std Dev': density_std
                    })
            
            df_group = pd.DataFrame(group_summary_rows)
            df_group.to_excel(writer, sheet_name='Group Summary', index=False)

    return results_dir

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_analysis.py path/to/image.nd2 path/to/rois.json")
        sys.exit(1)
    
    process_nd2_with_rois(sys.argv[1], sys.argv[2]) 