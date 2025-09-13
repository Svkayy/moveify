#!/usr/bin/env python3
"""
Dance Sync Analysis - GUI Version
Allows users to select videos from their laptop using a file picker
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import subprocess
import sys
from pathlib import Path

class DanceSyncGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dance Sync Analysis")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Variables to store selected video paths
        self.video1_path = tk.StringVar()
        self.video2_path = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="Dance Sync Analysis", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Compare your dance moves to a model dancer's performance",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Video 1 Selection
        video1_frame = tk.Frame(self.root, bg='#f0f0f0')
        video1_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(
            video1_frame, 
            text="Your Dance Video:", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        video1_entry_frame = tk.Frame(video1_frame, bg='#f0f0f0')
        video1_entry_frame.pack(fill='x', pady=(5, 0))
        
        self.video1_entry = tk.Entry(
            video1_entry_frame, 
            textvariable=self.video1_path,
            font=("Arial", 10),
            state='readonly',
            width=50
        )
        self.video1_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        video1_button = tk.Button(
            video1_entry_frame,
            text="Browse",
            command=self.select_video1,
            bg='#3498db',
            fg='white',
            font=("Arial", 10, "bold"),
            padx=20,
            pady=5
        )
        video1_button.pack(side='right')
        
        # Video 2 Selection
        video2_frame = tk.Frame(self.root, bg='#f0f0f0')
        video2_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(
            video2_frame, 
            text="Model Dance Video:", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        video2_entry_frame = tk.Frame(video2_frame, bg='#f0f0f0')
        video2_entry_frame.pack(fill='x', pady=(5, 0))
        
        self.video2_entry = tk.Entry(
            video2_entry_frame, 
            textvariable=self.video2_path,
            font=("Arial", 10),
            state='readonly',
            width=50
        )
        self.video2_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        video2_button = tk.Button(
            video2_entry_frame,
            text="Browse",
            command=self.select_video2,
            bg='#3498db',
            fg='white',
            font=("Arial", 10, "bold"),
            padx=20,
            pady=5
        )
        video2_button.pack(side='right')
        
        # Analysis Options
        options_frame = tk.LabelFrame(
            self.root, 
            text="Analysis Options", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        options_frame.pack(pady=20, padx=20, fill='x')
        
        # Output directory selection
        output_frame = tk.Frame(options_frame, bg='#f0f0f0')
        output_frame.pack(fill='x', pady=10, padx=10)
        
        tk.Label(
            output_frame,
            text="Output Directory:",
            font=("Arial", 10),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        self.output_dir = tk.StringVar(value=os.getcwd())
        output_entry_frame = tk.Frame(output_frame, bg='#f0f0f0')
        output_entry_frame.pack(fill='x', pady=(5, 0))
        
        output_entry = tk.Entry(
            output_entry_frame,
            textvariable=self.output_dir,
            font=("Arial", 10),
            width=50
        )
        output_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        output_button = tk.Button(
            output_entry_frame,
            text="Browse",
            command=self.select_output_dir,
            bg='#95a5a6',
            fg='white',
            font=("Arial", 10),
            padx=15,
            pady=2
        )
        output_button.pack(side='right')
        
        # Checkboxes for options
        self.create_video = tk.BooleanVar(value=True)
        self.create_report = tk.BooleanVar(value=True)
        self.create_visualization = tk.BooleanVar(value=True)
        
        tk.Checkbutton(
            options_frame,
            text="Create comparison video",
            variable=self.create_video,
            bg='#f0f0f0',
            font=("Arial", 10)
        ).pack(anchor='w', padx=10, pady=2)
        
        tk.Checkbutton(
            options_frame,
            text="Create detailed report (JSON)",
            variable=self.create_report,
            bg='#f0f0f0',
            font=("Arial", 10)
        ).pack(anchor='w', padx=10, pady=2)
        
        tk.Checkbutton(
            options_frame,
            text="Create score visualization",
            variable=self.create_visualization,
            bg='#f0f0f0',
            font=("Arial", 10)
        ).pack(anchor='w', padx=10, pady=2)
        
        # Progress bar
        self.progress_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.progress_frame.pack(pady=10, padx=20, fill='x')
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate'
        )
        self.progress_bar.pack(fill='x')
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="Ready to analyze",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#27ae60'
        )
        self.progress_label.pack(pady=(5, 0))
        
        # Analyze button
        analyze_button = tk.Button(
            self.root,
            text="Start Dance Sync Analysis",
            command=self.start_analysis,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 14, "bold"),
            padx=30,
            pady=10
        )
        analyze_button.pack(pady=20)
        
        # Status text
        self.status_text = tk.Text(
            self.root,
            height=8,
            width=70,
            font=("Courier", 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            state='disabled'
        )
        self.status_text.pack(pady=(0, 20), padx=20, fill='both', expand=True)
        
    def select_video1(self):
        """Select the first video file."""
        file_path = filedialog.askopenfilename(
            title="Select Your Dance Video",
            filetypes=[
                ("Video files", "*.mov *.mp4 *.avi *.mkv *.wmv"),
                ("MOV files", "*.mov"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.video1_path.set(file_path)
            self.log_message(f"Selected video 1: {os.path.basename(file_path)}")
    
    def select_video2(self):
        """Select the second video file."""
        file_path = filedialog.askopenfilename(
            title="Select Model Dance Video",
            filetypes=[
                ("Video files", "*.mov *.mp4 *.avi *.mkv *.wmv"),
                ("MOV files", "*.mov"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.video2_path.set(file_path)
            self.log_message(f"Selected video 2: {os.path.basename(file_path)}")
    
    def select_output_dir(self):
        """Select output directory."""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir.set(dir_path)
            self.log_message(f"Output directory: {dir_path}")
    
    def log_message(self, message):
        """Add a message to the status log."""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
        self.root.update()
    
    def start_analysis(self):
        """Start the dance sync analysis."""
        # Validate inputs
        if not self.video1_path.get():
            messagebox.showerror("Error", "Please select your dance video")
            return
        
        if not self.video2_path.get():
            messagebox.showerror("Error", "Please select the model dance video")
            return
        
        if not os.path.exists(self.video1_path.get()):
            messagebox.showerror("Error", f"Video 1 not found: {self.video1_path.get()}")
            return
        
        if not os.path.exists(self.video2_path.get()):
            messagebox.showerror("Error", f"Video 2 not found: {self.video2_path.get()}")
            return
        
        # Start analysis in a separate thread
        self.run_analysis()
    
    def run_analysis(self):
        """Run the dance sync analysis."""
        try:
            # Update UI
            self.progress_bar.start()
            self.progress_label.config(text="Starting analysis...", fg='#f39c12')
            self.log_message("=" * 50)
            self.log_message("DANCE SYNC ANALYSIS STARTED")
            self.log_message("=" * 50)
            
            # Prepare command
            cmd = [
                "python3", "dance.py",
                self.video1_path.get(),
                self.video2_path.get(),
                "--output-dir", self.output_dir.get()
            ]
            
            if not self.create_video.get():
                cmd.append("--no-video")
            
            self.log_message(f"Command: {' '.join(cmd)}")
            self.log_message("")
            
            # Change to the script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            
            # Activate virtual environment and run
            if os.path.exists("venv_mediapipe/bin/activate"):
                # Use the virtual environment
                full_cmd = f"source venv_mediapipe/bin/activate && {' '.join(cmd)}"
                process = subprocess.Popen(
                    full_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
            else:
                # Run directly
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
            
            # Read output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log_message(line.strip())
                    self.root.update()
            
            process.wait()
            
            # Check if analysis was successful
            if process.returncode == 0:
                self.progress_bar.stop()
                self.progress_label.config(text="Analysis completed successfully!", fg='#27ae60')
                self.log_message("")
                self.log_message("=" * 50)
                self.log_message("ANALYSIS COMPLETED SUCCESSFULLY!")
                self.log_message("=" * 50)
                
                # Show results
                output_files = []
                output_dir = self.output_dir.get()
                
                if self.create_video.get():
                    video_file = os.path.join(output_dir, "comparison_output.mp4")
                    if os.path.exists(video_file):
                        output_files.append(f"📹 Comparison video: {video_file}")
                
                if self.create_report.get():
                    report_file = os.path.join(output_dir, "dance_analysis_report.json")
                    if os.path.exists(report_file):
                        output_files.append(f"📊 Analysis report: {report_file}")
                
                if self.create_visualization.get():
                    viz_file = os.path.join(output_dir, "sync_scores.png")
                    if os.path.exists(viz_file):
                        output_files.append(f"📈 Score visualization: {viz_file}")
                
                if output_files:
                    self.log_message("")
                    self.log_message("Generated files:")
                    for file_info in output_files:
                        self.log_message(f"  {file_info}")
                
                messagebox.showinfo("Success", "Dance sync analysis completed successfully!")
            else:
                self.progress_bar.stop()
                self.progress_label.config(text="Analysis failed", fg='#e74c3c')
                self.log_message("")
                self.log_message("=" * 50)
                self.log_message("ANALYSIS FAILED!")
                self.log_message("=" * 50)
                messagebox.showerror("Error", "Analysis failed. Check the log for details.")
        
        except Exception as e:
            self.progress_bar.stop()
            self.progress_label.config(text="Analysis failed", fg='#e74c3c')
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = DanceSyncGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
