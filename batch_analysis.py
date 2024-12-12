import sys
from pathlib import Path
import logging
import json
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, QTableWidget,
                            QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt
from interactive_roi_selector import MainWindow as ROISelector
from run_analysis import process_nd2_with_rois
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_analysis.log')
    ]
)

class GroupAssignmentWindow(QMainWindow):
    def __init__(self, samples, data_folder, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.data_folder = data_folder
        self.group_assignments = {}
        # Define experiment-specific factors and their options
        self.factors = ['Treatment', 'IOP', 'AAV', 'Genotype', 'Other']
        self.factor_options = {
            'Treatment': ['NR', 'NAM', 'NMN', 'GLP-1RA', 'Control'],
            'IOP': ['Normal', 'High'],
            'AAV': ['Control', 'KD', 'OE'],
            'Genotype': ['Wild type', 'KO'],
            'Other': []  # Empty list for custom entries
        }
        self.initUI()
        
        # Center window on screen
        self.center_on_screen()
    
    def center_on_screen(self):
        """Center the window on the screen"""
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().screenGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
        
    def initUI(self):
        self.setWindowTitle('Assign Groups to Samples')
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add instructions
        instructions = QLabel(
            "Assign multiple factors to each sample. These will be used for grouping in the analysis."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Create factor management section
        factor_section = QHBoxLayout()
        
        # Factor input
        factor_input = QHBoxLayout()
        self.factor_entry = QComboBox()
        self.factor_entry.setEditable(True)
        self.factor_entry.addItems(self.factors)
        factor_input.addWidget(QLabel("New Factor:"))
        factor_input.addWidget(self.factor_entry)
        
        # Add factor button
        add_factor_btn = QPushButton("Add Factor")
        add_factor_btn.clicked.connect(self.add_factor)
        factor_input.addWidget(add_factor_btn)
        
        factor_section.addLayout(factor_input)
        layout.addLayout(factor_section)
        
        # Create table for sample-factor assignments
        self.table = QTableWidget()
        self.setup_table()
        layout.addWidget(self.table)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        confirm_btn = QPushButton('Confirm Assignments')
        confirm_btn.clicked.connect(self.confirm_assignments)
        button_layout.addWidget(confirm_btn)
        
        layout.addLayout(button_layout)
    
    def setup_table(self):
        """Setup the table with current factors"""
        self.table.clear()
        self.table.setColumnCount(len(self.factors) + 1)  # +1 for sample name
        headers = ['Sample'] + self.factors
        self.table.setHorizontalHeaderLabels(headers)
        
        # Set column widths
        header = self.table.horizontalHeader()
        for i in range(self.table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        # Add rows for each sample
        self.table.setRowCount(len(self.samples))
        for i, sample in enumerate(self.samples):
            # Sample name (read-only)
            sample_item = QTableWidgetItem(sample)
            sample_item.setFlags(sample_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, sample_item)
            
            # Add combo boxes for each factor
            for j, factor in enumerate(self.factors, 1):
                combo = QComboBox()
                if factor in self.factor_options:
                    combo.addItems(self.factor_options[factor])
                combo.setEditable(True)
                self.table.setCellWidget(i, j, combo)
    
    def add_factor(self):
        """Add a new factor column"""
        new_factor = self.factor_entry.currentText()
        if new_factor and new_factor not in self.factors:
            self.factors.append(new_factor)
            self.factor_options[new_factor] = []  # Initialize empty options
            self.setup_table()
    
    def confirm_assignments(self):
        """Collect all factor assignments and save to Excel"""
        # Collect assignments as before
        for row in range(self.table.rowCount()):
            sample = self.table.item(row, 0).text()
            sample_factors = {}
            
            # Collect all factor values
            for col, factor in enumerate(self.factors, 1):
                combo = self.table.cellWidget(row, col)
                value = combo.currentText()
                sample_factors[factor] = value
                
                # Add to factor options if not already present
                if value not in self.factor_options[factor]:
                    self.factor_options[factor].append(value)
            
            self.group_assignments[sample] = sample_factors
        
        # Create DataFrame for Excel export
        data = []
        for sample, factors in self.group_assignments.items():
            row_data = {'Sample': sample}
            row_data.update(factors)
            data.append(row_data)
        
        df = pd.DataFrame(data)
        
        # Save to Excel
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = Path(self.data_folder) / f'sample_groups_{timestamp}.xlsx'
            df.to_excel(excel_path, index=False)
            logging.info(f"Saved group assignments to {excel_path}")
            
            # Show success message
            QMessageBox.information(
                self,
                "Success",
                "Group assignments saved successfully.\nProceeding with analysis...",
                QMessageBox.Ok
            )
            
            # Set completion flag and close window
            self.assignment_completed = True
            self.close()
            QApplication.processEvents()  # Process any pending events
            
        except Exception as e:
            logging.error(f"Error saving group assignments to Excel: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error saving group assignments: {str(e)}",
                QMessageBox.Ok
            )
            return

class BatchAnalyzer:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.results_folder = self.data_folder / 'analysis_results'
        self.results_folder.mkdir(exist_ok=True)
        self.sample_groups = {}  # Store sample group assignments
        
    def find_nd2_files(self):
        """Find all .nd2 files in the data folder"""
        return list(self.data_folder.glob('*.nd2'))
        
    def find_roi_files(self):
        """Find all ROI JSON files in the data folder"""
        return list(self.data_folder.glob('*_rois_*.json'))
        
    def match_nd2_to_rois(self):
        """Match ND2 files with their corresponding ROI files"""
        nd2_files = self.find_nd2_files()
        roi_files = self.find_roi_files()
        
        matches = []
        unmatched_nd2 = []
        
        for nd2_file in nd2_files:
            # Find ROI files that start with the ND2 filename
            matching_rois = [
                roi for roi in roi_files 
                if roi.stem.startswith(nd2_file.stem)
            ]
            
            if matching_rois:
                # Use the most recent ROI file if multiple exist
                latest_roi = max(matching_rois, key=lambda x: x.stat().st_mtime)
                matches.append((nd2_file, latest_roi))
            else:
                unmatched_nd2.append(nd2_file)
                
        return matches, unmatched_nd2
        
    def roi_selection_mode(self):
        """Interactive ROI selection for all ND2 files"""
        nd2_files = self.find_nd2_files()
        
        # Use existing QApplication instance if available
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Track windows to prevent premature garbage collection
        windows = []
        
        try:
            for nd2_file in nd2_files:
                logging.info(f"Processing {nd2_file.name}")
                
                # Check if ROI file already exists
                existing_rois = list(self.data_folder.glob(f"{nd2_file.stem}_rois_*.json"))
                if existing_rois:
                    # Show notification about existing ROIs
                    QMessageBox.information(
                        None,
                        "ROIs Exist",
                        f"ROIs already exist for {nd2_file.name}\nSkipping to next file...",
                        QMessageBox.Ok
                    )
                    logging.info(f"ROI file already exists for {nd2_file.name}")
                    continue
                
                try:
                    # Create window in batch mode
                    window = ROISelector(str(nd2_file), batch_mode=True)
                    window.show()
                    windows.append(window)
                    
                    # Process events until window is closed
                    while window.isVisible():
                        app.processEvents()
                    
                except Exception as e:
                    logging.error(f"Error processing {nd2_file.name}: {str(e)}")
                    continue
                
            logging.info("ROI selection completed for all files")
            
            # Show completion message
            QMessageBox.information(
                None,
                "Batch ROI Selection Complete",
                "ROI selection has been completed for all files.",
                QMessageBox.Ok
            )
            
        finally:
            # Clean up windows
            for window in windows:
                window.close()
        
    def analyze_all_samples(self):
        """Run analysis on all matched ND2-ROI pairs"""
        matches, unmatched = self.match_nd2_to_rois()
        
        if unmatched:
            logging.warning(f"Found {len(unmatched)} ND2 files without ROIs: "
                          f"{', '.join(str(f.name) for f in unmatched)}")
        
        results = []
        for nd2_file, roi_file in matches:
            logging.info(f"Processing {nd2_file.name} with ROIs from {roi_file.name}")
            
            try:
                # Get group information for this sample
                sample_name = nd2_file.stem
                group_info = self.sample_groups.get(sample_name, None)
                
                # Process the pair with group information
                results_dir = process_nd2_with_rois(
                    str(nd2_file),
                    str(roi_file),
                    str(self.results_folder / nd2_file.stem),
                    group_info=group_info
                )
                
                results.append({
                    'sample': sample_name,
                    'nd2_file': str(nd2_file),
                    'roi_file': str(roi_file),
                    'results_dir': str(results_dir),
                    'group_info': group_info
                })
                
            except Exception as e:
                logging.error(f"Error processing {nd2_file.name}: {str(e)}")
                continue
                
        return results
        
    def assign_groups(self):
        """Open window for group assignment"""
        matches, _ = self.match_nd2_to_rois()
        samples = [nd2_file.stem for nd2_file, _ in matches]
        
        if not samples:
            logging.warning("No samples found for group assignment")
            return False
        
        # Check for existing group assignments
        group_file = self.data_folder / 'sample_groups.json'
        if group_file.exists():
            with open(group_file) as f:
                existing_groups = json.load(f)
            
            # Show dialog with existing assignments
            msg = QMessageBox()
            msg.setWindowTitle("Existing Group Assignments")
            msg.setText("Found existing group assignments:")
            
            # Create detailed text showing current assignments
            details = "\n".join([
                f"Sample: {sample}\n" + \
                "\n".join(f"  {factor}: {value}" for factor, value in factors.items()) + "\n"
                for sample, factors in existing_groups.items()
            ])
            
            msg.setDetailedText(details)
            msg.setIcon(QMessageBox.Information)
            
            # Add custom buttons
            keep_btn = msg.addButton("Keep Existing", QMessageBox.AcceptRole)
            reassign_btn = msg.addButton("Reassign Groups", QMessageBox.ResetRole)
            cancel_btn = msg.addButton(QMessageBox.Cancel)
            
            msg.exec_()
            
            clicked_button = msg.clickedButton()
            if clicked_button == keep_btn:
                self.sample_groups = existing_groups
                return True
            elif clicked_button == cancel_btn:
                return False
            # If reassign_btn, continue with new assignment
        
        app = QApplication.instance() or QApplication(sys.argv)
        window = GroupAssignmentWindow(samples, str(self.data_folder))
        
        # Set parent window if available
        if hasattr(self, 'parent_window'):
            window.setParent(self.parent_window)
            window.setWindowFlags(Qt.Window)
        
        # Initialize completion flag
        window.assignment_completed = False
        window.show()
        
        # Wait for window to close
        while window.isVisible():
            app.processEvents()
        
        # Check if assignments were made and confirmed
        if not window.assignment_completed or not window.group_assignments:
            return False
        
        self.sample_groups = window.group_assignments
        
        # Save group assignments
        with open(group_file, 'w') as f:
            json.dump(self.sample_groups, f, indent=4)
        
        return True
        
    def generate_summary_report(self, analysis_results):
        """Generate final summary report of all analyses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.results_folder / f"summary_report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for better organization
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Create specific figure subdirectories
        density_plots_dir = figures_dir / "density_plots"
        group_plots_dir = figures_dir / "group_plots"
        interaction_plots_dir = figures_dir / "interaction_plots"
        
        density_plots_dir.mkdir(exist_ok=True)
        group_plots_dir.mkdir(exist_ok=True)
        interaction_plots_dir.mkdir(exist_ok=True)
        
        # Load group assignments if they exist
        group_file = self.data_folder / 'sample_groups.json'
        if group_file.exists():
            with open(group_file) as f:
                self.sample_groups = json.load(f)
        
        # Collect all Excel files
        all_data = []
        group_data = []
        
        for result in analysis_results:
            results_dir = Path(result['results_dir'])
            excel_file = next(results_dir.glob("*_analysis_summary.xlsx"))
            
            # Read both sheets
            df_summary = pd.read_excel(excel_file, sheet_name='ROI Summary')
            df_group = pd.read_excel(excel_file, sheet_name='Group Summary')
            
            # Add sample information
            sample_name = result['sample']
            df_summary['Sample'] = sample_name
            df_group['Sample'] = sample_name
            
            # Add sample group factors if available
            if sample_name in self.sample_groups:
                for factor, value in self.sample_groups[sample_name].items():
                    df_summary[f'Sample Group.{factor}'] = value
                    df_group[f'Sample Group.{factor}'] = value
            
            all_data.append(df_summary)
            group_data.append(df_group)
        
        # Combine all data
        combined_summary = pd.concat(all_data, ignore_index=True)
        combined_groups = pd.concat(group_data, ignore_index=True)
        
        # Save combined data
        excel_path = report_dir / f"combined_analysis_summary_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            combined_summary.to_excel(writer, sheet_name='All ROIs', index=False)
            combined_groups.to_excel(writer, sheet_name='Group Summaries', index=False)
        
        # Generate visualizations with organized directories
        self.generate_summary_plots(combined_summary, combined_groups, 
                                  density_plots_dir, group_plots_dir, interaction_plots_dir)
        
        # Create summary text report
        self.create_text_summary(combined_summary, combined_groups, report_dir)
        
        return report_dir
        
    def generate_summary_plots(self, roi_data, group_data, density_dir, group_dir, interaction_dir):
        """Generate summary plots for the report"""
        import seaborn as sns
        sns.set_style("whitegrid")
        
        # Basic density plots in density_dir
        # 1. Box plot
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(data=roi_data, x='Group', y='Cell Density (mm²)')
        plt.title('Cell Density Distribution by Group')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(density_dir / 'density_by_group_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Bar plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=group_data, x='Sample', y='Average Density (cells/mm²)')
        plt.title('Average Cell Density by Sample')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(density_dir / 'density_by_sample_barplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Heatmap
        pivot_data = roi_data.pivot_table(
            values='Cell Density (mm²)',
            index='Sample',
            columns='Group',
            aggfunc='mean'
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, 
                    annot=True, 
                    fmt='.0f', 
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Cells/mm²'})
        plt.title('Cell Density Heatmap by Group and Sample')
        plt.tight_layout()
        plt.savefig(density_dir / 'density_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Violin plot
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(data=roi_data, x='Group', y='Cell Density (mm²)')
        plt.title('Cell Density Distribution by Group (Violin Plot)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(density_dir / 'density_violin_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Swarm plot
        plt.figure(figsize=(12, 6))
        ax = sns.swarmplot(data=roi_data, x='Group', y='Cell Density (mm²)', size=5)
        plt.title('Individual ROI Cell Densities by Group')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(density_dir / 'density_swarm_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Group-specific plots in group_dir
        if any(col.startswith('Sample Group.') for col in roi_data.columns):
            factors = [col.replace('Sample Group.', '') 
                      for col in roi_data.columns if col.startswith('Sample Group.')]
            
            for factor in factors:
                factor_col = f'Sample Group.{factor}'
                
                # Box plot by factor
                plt.figure(figsize=(12, 6))
                ax = sns.boxplot(data=roi_data, x=factor_col, y='Cell Density (mm²)')
                plt.title(f'Cell Density Distribution by {factor}')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(group_dir / f'density_by_{factor.lower()}_boxplot.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Violin plot by factor
                plt.figure(figsize=(12, 6))
                ax = sns.violinplot(data=roi_data, x=factor_col, y='Cell Density (mm²)')
                plt.title(f'Cell Density Distribution by {factor}')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(group_dir / f'density_by_{factor.lower()}_violin.png',
                           dpi=300, bbox_inches='tight')
                plt.close()

                # Interaction plots in interaction_dir
                if len(factors) > 1:
                    for other_factor in factors:
                        if other_factor != factor:
                            other_col = f'Sample Group.{other_factor}'
                            
                            # Box plot with interaction
                            plt.figure(figsize=(14, 7))
                            ax = sns.boxplot(data=roi_data, 
                                           x=factor_col, 
                                           y='Cell Density (mm²)',
                                           hue=other_col)
                            plt.title(f'Cell Density by {factor} and {other_factor}')
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                            plt.legend(title=other_factor, bbox_to_anchor=(1.05, 1))
                            plt.tight_layout()
                            plt.savefig(
                                interaction_dir / f'density_{factor.lower()}_{other_factor.lower()}_interaction.png',
                                dpi=300, bbox_inches='tight'
                            )
                            plt.close()
                            
                            # Violin plot with interaction
                            plt.figure(figsize=(14, 7))
                            ax = sns.violinplot(data=roi_data, 
                                              x=factor_col, 
                                              y='Cell Density (mm²)',
                                              hue=other_col)
                            plt.title(f'Cell Density Distribution by {factor} and {other_factor}')
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                            plt.legend(title=other_factor, bbox_to_anchor=(1.05, 1))
                            plt.tight_layout()
                            plt.savefig(
                                interaction_dir / f'density_{factor.lower()}_{other_factor.lower()}_violin.png',
                                dpi=300, bbox_inches='tight'
                            )
                            plt.close()
    
    def create_text_summary(self, roi_data, group_data, report_dir):
        """Create text summary of the analysis"""
        summary_lines = [
            "Batch Analysis Summary Report",
            "=" * 30,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of samples analyzed: {len(roi_data['Sample'].unique())}",
            "\nOverall Statistics:",
            f"Total ROIs analyzed: {len(roi_data)}",
            f"Average cell density: {roi_data['Cell Density (mm²)'].mean():.1f} cells/mm²",
            f"Density range: {roi_data['Cell Density (mm²)'].min():.1f} - {roi_data['Cell Density (mm²)'].max():.1f} cells/mm²",
            "\nGroup Statistics:",
        ]
        
        # Add group statistics
        for group in roi_data['Group'].unique():
            group_stats = roi_data[roi_data['Group'] == group]['Cell Density (mm²)']
            summary_lines.extend([
                f"\n{group}:",
                f"  Mean density: {group_stats.mean():.1f} cells/mm²",
                f"  Std deviation: {group_stats.std():.1f}",
                f"  Number of ROIs: {len(group_stats)}"
            ])
        
        # Write summary to file
        with open(report_dir / 'analysis_summary.txt', 'w') as f:
            f.write('\n'.join(summary_lines))

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_analysis.py path/to/data/folder")
        sys.exit(1)
        
    data_folder = sys.argv[1]
    analyzer = BatchAnalyzer(data_folder)
    
    print("\nBatch Analysis Options:")
    print("1. ROI Selection Mode (Select ROIs for all ND2 files)")
    print("2. Analysis Mode (Run analysis on all ND2 files with existing ROIs)")
    print("3. Full Pipeline (ROI Selection followed by Analysis)")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        analyzer.roi_selection_mode()
    elif choice == '2':
        # Add group assignment before analysis
        analyzer.assign_groups()
        results = analyzer.analyze_all_samples()
        report_dir = analyzer.generate_summary_report(results)
        print(f"\nAnalysis complete! Summary report saved in: {report_dir}")
    elif choice == '3':
        analyzer.roi_selection_mode()
        analyzer.assign_groups()
        results = analyzer.analyze_all_samples()
        report_dir = analyzer.generate_summary_report(results)
        print(f"\nFull pipeline complete! Summary report saved in: {report_dir}")
    else:
        print("Invalid choice!")

if __name__ == '__main__':
    main() 