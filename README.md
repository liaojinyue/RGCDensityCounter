# RGC Density Counter

A sophisticated tool for analyzing Retinal Ganglion Cell (RGC) density with advanced cell segmentation, ROI selection, and editing capabilities.

## Features

- **Advanced Cell Segmentation**
  - Automated cell detection using Cellpose
  - Custom RGC model support
  - GPU acceleration when available
  - Batch processing capabilities

- **Interactive ROI Selection**
  - Anatomically-guided ROI placement
  - Reference circles for standardized measurements
  - Support for multiple retinal regions (Superior, Inferior, Temporal, Nasal)
  - Automatic cell counting within ROIs

- **Cell Mask Editing**
  - Traditional editor with add/delete functionality
  - Modern napari-based editor for enhanced visualization
  - Undo/redo support
  - ROI-specific editing

- **Comprehensive Analysis**
  - Automated cell density calculations
  - Region-specific quantification
  - Excel report generation
  - Statistical summaries
  - Visualization exports

## Installation

1. Create a new conda environment:



## Usage

1. **Launch the Application**

2. **Process Images**
   - Use "Process Single Image" for individual files
   - Use "Batch Process Images" for multiple files
   - Select ND2 files when prompted

3. **Select ROIs**
   - Click "Select ROIs" after processing
   - Set retina center for reference circles
   - Draw ROIs using Shift+Click
   - Assign anatomical regions

4. **Edit Cell Masks**
   - Use traditional editor or napari editor
   - Delete incorrect cells
   - Add missed cells
   - Save changes

5. **Generate Reports**
   - Automatic Excel report generation
   - Cell counts per ROI
   - Density calculations
   - Statistical summaries

## File Structure

- `retinal_analysis_new.py`: Main application file
- `rect_roi_selector.py`: ROI selection implementation
- `cell_mask_editor.py`: Traditional cell mask editor
- `napari_cell_editor.py`: Modern napari-based editor
- `nd2handler.py`: ND2 file handling utilities

## Data Organization

The tool automatically organizes data in the following structure:


## Requirements

- Python 3.9+
- PyQt5
- numpy
- scipy
- scikit-image
- cellpose
- torch
- napari
- pandas
- nd2reader

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Cellpose for cell segmentation
- napari for advanced visualization
- The scientific Python community for essential tools

## Support

For support, please open an issue in the GitHub repository or contact [your contact information].
