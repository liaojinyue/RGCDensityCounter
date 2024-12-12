from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                           QMessageBox, QFrame, QProgressDialog)
from PyQt5.QtCore import Qt, QTimer
from interactive_roi_selector import MainWindow as ROISelector
from run_analysis import process_nd2_with_rois
from pathlib import Path
import sys
import logging
from batch_analysis import BatchAnalyzer

class RetinalAnalysisTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.roi_window = None  # Store reference to ROI window
        self.batch_analyzer = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('RGC Density Counter')
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Add title
        title = QLabel('RGC Density Counter')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 24px; font-weight: bold; margin: 20px;')
        main_layout.addWidget(title)
        
        # Create file selection section
        file_frame = QFrame()
        file_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        file_layout = QVBoxLayout(file_frame)
        
        file_label = QLabel('Selected ND2 File:')
        self.file_path_label = QLabel('No file selected')
        self.file_path_label.setStyleSheet('color: gray; font-style: italic;')
        
        select_file_btn = QPushButton('Select ND2 File')
        select_file_btn.clicked.connect(self.select_file)
        select_file_btn.setStyleSheet('font-size: 14px; padding: 8px;')
        
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(select_file_btn)
        
        main_layout.addWidget(file_frame)
        
        # Add action buttons
        buttons_layout = QHBoxLayout()
        
        # ROI Selection button
        roi_btn = QPushButton('ROI Selection and analysis\n')
        roi_btn.clicked.connect(self.open_roi_selector)
        roi_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 15px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # Analysis button
        analysis_btn = QPushButton('Run Analysis with existing mask\n(1. Select nd2 file, 2. Select ROI file)')
        analysis_btn.clicked.connect(self.run_analysis)
        analysis_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 15px;
                background-color: #008CBA;
                color: white;
                border-radius: 5px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #007399;
            }
        """)
        
        # Add tooltip
        analysis_btn.setToolTip("Select a previously saved ROI file (.json) to analyze")
        
        buttons_layout.addWidget(roi_btn)
        buttons_layout.addWidget(analysis_btn)
        main_layout.addLayout(buttons_layout)
        
        # Add instructions
        instructions = QLabel(
            "\nInstructions:\n\n"
            "1. Select an ND2 file using the button above\n\n"
            "2. Choose one of the following options:\n"
            "   • ROI Selection: Open the file to select regions of interest\n"
            "   • Run Analysis: Process the file with existing ROIs\n\n"
        )
        instructions.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        main_layout.addWidget(instructions)
        
        # Store current file path
        self.current_file = None
        
        # Set window size and position
        self.setGeometry(100, 100, 600, 600)
        
        # Add batch analysis section
        batch_frame = QFrame()
        batch_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        batch_layout = QVBoxLayout(batch_frame)
        
        batch_label = QLabel('If you have multiple ND2 files:')
        batch_label.setStyleSheet('font-weight: bold;')
        batch_layout.addWidget(batch_label)
        
        # Single batch analysis button
        batch_btn = QPushButton('Batch Analysis')
        batch_btn.clicked.connect(self.show_batch_options)
        batch_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        batch_layout.addWidget(batch_btn)
        
        main_layout.addWidget(batch_frame)
        
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select single ND2 File",
            "",
            "ND2 Files (*.nd2)"
        )
        
        if file_path:
            self.current_file = file_path
            self.file_path_label.setText(Path(file_path).name)
            self.file_path_label.setStyleSheet('color: black; font-style: normal;')
            
    def show_progress(self, title, message):
        """Show progress dialog"""
        progress = QProgressDialog(message, None, 0, 0, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setAutoClose(True)
        progress.show()
        QApplication.processEvents()
        return progress

    def open_roi_selector(self):
        if not self.current_file:
            QMessageBox.warning(self, "No File Selected", "Please select an ND2 file first.")
            return
            
        try:
            # Show progress dialog
            progress = self.show_progress("Loading Image", "Loading ND2 file...")
            QApplication.processEvents()  # Ensure UI updates
            
            # Create and show ROI selector
            if self.roi_window is None:  # Only create if not exists
                self.roi_window = ROISelector(self.current_file)
                self.roi_window.parent_tool = self
            
            progress.close()
            self.roi_window.show()
            self.roi_window.raise_()  # Bring window to front
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Error", f"Error opening file: {str(e)}")

    def run_analysis(self):
        """Manual analysis with file selection"""
        try:
            # Get ND2 file if not already selected
            if not self.current_file:
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select ND2 File",
                    "",
                    "ND2 Files (*.nd2)"
                )
                
                if not file_path:
                    return
                    
                self.current_file = file_path
                self.file_path_label.setText(Path(file_path).name)
                self.file_path_label.setStyleSheet('color: black; font-style: normal;')
            
            # Get ROI JSON file
            roi_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select ROI File",
                str(Path(self.current_file).parent),
                "JSON Files (*.json)"
            )
            
            if not roi_path:
                return
                
            progress = self.show_progress("Analyzing", "Processing image and ROIs...")
            
            # Auto-generate save directory
            save_dir = Path(self.current_file).parent / 'analysis_results'
            save_dir.mkdir(exist_ok=True)
            
            # Run analysis
            results_dir = process_nd2_with_rois(self.current_file, roi_path, str(save_dir))
            
            progress.close()
            
            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Analysis of {Path(self.current_file).name} complete!\n\n"
                f"Results saved in:\n{results_dir}"
            )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")

    def run_analysis_with_roi(self, roi_path):
        """Run analysis with automatically saved ROI file"""
        try:
            progress = self.show_progress("Analyzing", "Processing image and ROIs...")
            
            # Auto-generate save directory
            save_dir = Path(self.current_file).parent / 'analysis_results'
            save_dir.mkdir(exist_ok=True)
            
            # Run analysis
            results_dir = process_nd2_with_rois(self.current_file, roi_path, str(save_dir))
            
            progress.close()
            
            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Analysis of {Path(self.current_file).name} complete!\n\n"
                f"Results saved in:\n{results_dir}"
            )
            
            # Close window after 1 second
            QTimer.singleShot(1000, self.roi_window.close)
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")

    def create_buttons(self):
        """Create and style buttons"""
        # Create buttons
        self.roi_button = QPushButton("ROIs Selection and Analysis")  # Updated text
        self.analysis_button = QPushButton("Analysis with existing mask (select nd2 and mask file)")  # Updated text
        
        # Style buttons
        button_style = """
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        
        self.roi_button.setStyleSheet(button_style)
        self.analysis_button.setStyleSheet(button_style)
        
        # Add buttons to layout
        self.layout.addWidget(self.roi_button)
        self.layout.addWidget(self.analysis_button)
        
        # Connect buttons to functions
        self.roi_button.clicked.connect(self.start_roi_selection)
        self.analysis_button.clicked.connect(self.run_analysis_with_existing)

    def select_data_folder(self):
        """Select folder containing multiple ND2 files"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            # Create BatchAnalyzer with self as parent
            self.batch_analyzer = BatchAnalyzer(folder)
            # Set the parent window for group assignment
            self.batch_analyzer.parent_window = self
            return folder
        return None
    
    def batch_roi_selection(self):
        """Start batch ROI selection"""
        folder = self.select_data_folder()
        if folder:
            try:
                # Show progress dialog
                progress = self.show_progress("Initializing", "Starting batch ROI selection...")
                QApplication.processEvents()
                
                # Run batch ROI selection
                self.batch_analyzer.roi_selection_mode()
                
                progress.close()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error during batch ROI selection: {str(e)}")
                if 'progress' in locals():
                    progress.close()
    
    def batch_analysis(self):
        """Run batch analysis on existing ROIs"""
        folder = self.select_data_folder()
        if folder:
            try:
                # First assign groups (without progress dialog)
                if not self.batch_analyzer.assign_groups():
                    # If group assignment was cancelled or failed
                    return
                
                # After group assignment is complete and successful, show progress dialog
                progress = self.show_progress("Initializing", "Starting batch analysis...")
                QApplication.processEvents()
                
                # Run analysis
                results = self.batch_analyzer.analyze_all_samples()
                
                # Update progress
                progress.setLabelText("Generating summary report...")
                QApplication.processEvents()
                
                # Generate report
                report_dir = self.batch_analyzer.generate_summary_report(results)
                
                progress.close()
                
                # Show completion message
                QMessageBox.information(
                    self,
                    "Analysis Complete",
                    f"Batch analysis complete!\n\nResults saved in:\n{report_dir}"
                )
                
            except Exception as e:
                if 'progress' in locals():
                    progress.close()
                QMessageBox.critical(self, "Error", f"Error during batch analysis: {str(e)}")
    
    def full_batch_pipeline(self):
        """Run full batch pipeline"""
        folder = self.select_data_folder()
        if folder:
            try:
                # ROI selection first
                progress = self.show_progress("Initializing", "Starting ROI selection...")
                QApplication.processEvents()
                self.batch_analyzer.roi_selection_mode()
                progress.close()
                
                # Then group assignment (without progress dialog)
                if not self.batch_analyzer.assign_groups():
                    # If group assignment was cancelled or failed
                    return
                
                # After group assignment, show progress for analysis
                progress = self.show_progress("Processing", "Running analysis...")
                QApplication.processEvents()
                
                # Analysis
                results = self.batch_analyzer.analyze_all_samples()
                
                # Update progress
                progress.setLabelText("Generating summary report...")
                QApplication.processEvents()
                
                # Generate report
                report_dir = self.batch_analyzer.generate_summary_report(results)
                
                progress.close()
                
                # Show completion message
                QMessageBox.information(
                    self,
                    "Pipeline Complete",
                    f"Full batch pipeline complete!\n\nResults saved in:\n{report_dir}"
                )
                
            except Exception as e:
                if 'progress' in locals():
                    progress.close()
                QMessageBox.critical(self, "Error", f"Error during batch pipeline: {str(e)}")

    def show_batch_options(self):
        """Show dialog with batch analysis options"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch Analysis Options")
        msg.setText("Put all ND2 files in one folder and select the folder:")
        msg.setIcon(QMessageBox.Question)
        
        # Add custom buttons with descriptive text
        roi_button = msg.addButton("ROI Selection Mode\n(Select ROIs for all ND2 files)", 
                                  QMessageBox.ActionRole)
        analysis_button = msg.addButton("Analysis Mode\n(Run analysis with existing ROIs)", 
                                       QMessageBox.ActionRole)
        pipeline_button = msg.addButton("Full Pipeline\n(ROI Selection followed by Analysis)", 
                                       QMessageBox.ActionRole)
        cancel_button = msg.addButton(QMessageBox.Cancel)
        
        # Style the buttons
        button_style = """
            QPushButton {
                min-width: 200px;
                padding: 10px;
                text-align: left;
            }
        """
        for button in [roi_button, analysis_button, pipeline_button]:
            button.setStyleSheet(button_style)
        
        msg.exec_()
        
        # Handle button clicks
        clicked_button = msg.clickedButton()
        if clicked_button == roi_button:
            self.batch_roi_selection()
        elif clicked_button == analysis_button:
            self.batch_analysis()
        elif clicked_button == pipeline_button:
            self.full_batch_pipeline()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = RetinalAnalysisTool()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 