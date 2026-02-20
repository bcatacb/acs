#!/usr/bin/env python3
"""
Sound Bank Ingestion GUI
========================
Simple graphical tool for ingesting samples into the Sound Bank.

No command-line arguments needed - just click directories and run!
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
from soundbank.ingest import IngestionEngine


class SoundBankIngestionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Bank Ingestion Tool")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="./output")
        self.category = tk.StringVar(value="loops")
        self.normalize_mode = tk.StringVar(value="rms")
        self.target_rms = tk.DoubleVar(value=0.1)
        self.notch_db = tk.DoubleVar(value=-12.0)
        self.apply_notch = tk.BooleanVar(value=True)
        self.recursive = tk.BooleanVar(value=True)
        self.is_processing = False
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Title
        title = tk.Label(
            self.root,
            text="Sound Bank Ingestion Tool",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # =====================================================================
        # INPUT DIRECTORY
        # =====================================================================
        input_frame = ttk.LabelFrame(main_frame, text="1. Select Input Directory", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Big Browse button
        ttk.Button(
            input_frame,
            text="📁 CLICK HERE TO SELECT AUDIO FOLDER",
            command=self._select_input_dir,
            width=40
        ).pack(pady=10)
        
        # Show selected path
        self.input_label = tk.Label(input_frame, text="No folder selected", fg="gray", font=("Arial", 10, "bold"), wraplength=400)
        self.input_label.pack(anchor=tk.W, pady=5)
        
        self.input_dir.trace("w", lambda *args: self._update_input_label())
        
        # =====================================================================
        # OUTPUT DIRECTORY
        # =====================================================================
        output_frame = ttk.LabelFrame(main_frame, text="2. Output Directory", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        # Big Browse button
        ttk.Button(
            output_frame,
            text="📁 CLICK HERE TO SELECT OUTPUT FOLDER",
            command=self._select_output_dir,
            width=40
        ).pack(pady=10)
        
        # Show selected path
        self.output_label = tk.Label(output_frame, text="Default: ./output", fg="green", font=("Arial", 10, "bold"), wraplength=400)
        self.output_label.pack(anchor=tk.W, pady=5)
        
        self.output_dir.trace("w", lambda *args: self._update_output_label())
        
        # =====================================================================
        # CATEGORY
        # =====================================================================
        category_frame = ttk.LabelFrame(main_frame, text="3. Sample Category", padding="10")
        category_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(category_frame, text="What type of samples are these?").pack(anchor=tk.W, pady=(0, 5))
        
        for cat in ["808", "snare", "loops", "atmospheres"]:
            ttk.Radiobutton(
                category_frame,
                text=cat.capitalize(),
                variable=self.category,
                value=cat
            ).pack(anchor=tk.W)
        
        # =====================================================================
        # PROCESSING OPTIONS
        # =====================================================================
        options_frame = ttk.LabelFrame(main_frame, text="4. Processing Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        
        # Recursive option
        ttk.Checkbutton(
            options_frame,
            text="Search subdirectories",
            variable=self.recursive
        ).pack(anchor=tk.W, pady=2)
        
        # Spectral notch
        ttk.Checkbutton(
            options_frame,
            text="Apply spectral notch filter (preserve vocal pocket)",
            variable=self.apply_notch
        ).pack(anchor=tk.W, pady=2)
        
        # Normalization
        ttk.Label(options_frame, text="Normalization:").pack(anchor=tk.W, pady=(10, 5))
        for mode in ["rms", "peak", "none"]:
            ttk.Radiobutton(
                options_frame,
                text=mode.capitalize(),
                variable=self.normalize_mode,
                value=mode
            ).pack(anchor=tk.W, padx=20)
        
        # Target RMS
        rms_frame = ttk.Frame(options_frame)
        rms_frame.pack(fill=tk.X, pady=5, padx=20)
        ttk.Label(rms_frame, text="Target RMS (0.05-0.3):").pack(side=tk.LEFT)
        tk.Spinbox(
            rms_frame,
            from_=0.05,
            to=0.3,
            increment=0.01,
            textvariable=self.target_rms,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # =====================================================================
        # PROGRESS
        # =====================================================================
        progress_frame = ttk.LabelFrame(main_frame, text="5. Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.progress_text = tk.Text(progress_frame, height=8, width=80, font=("Courier", 9))
        self.progress_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=self.progress_text.yview)
        self.progress_text.config(yscrollcommand=scrollbar.set)
        
        # =====================================================================
        # BUTTONS
        # =====================================================================
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.ingest_button = ttk.Button(
            button_frame,
            text="Start Ingestion",
            command=self._start_ingestion
        )
        self.ingest_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear Log",
            command=lambda: self.progress_text.delete("1.0", tk.END)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Exit",
            command=self.root.quit
        ).pack(side=tk.RIGHT, padx=5)
    
    def _select_input_dir(self):
        """Open directory browser for input."""
        directory = filedialog.askdirectory(
            title="Select Input Directory (audio files to ingest)",
            initialdir=str(Path.cwd())
        )
        if directory:
            self.input_dir.set(directory)
    
    def _select_output_dir(self):
        """Open directory browser for output."""
        directory = filedialog.askdirectory(
            title="Select Output Directory (for master_bank.wav and bank.db)",
            initialdir=str(Path.cwd())
        )
        if directory:
            self.output_dir.set(directory)
    
    def _update_input_label(self):
        """Update input directory label."""
        if self.input_dir.get():
            self.input_label.config(text=f"✓ Selected: {self.input_dir.get()}", fg="green")
        else:
            self.input_label.config(text="No folder selected", fg="gray")
    
    def _update_output_label(self):
        """Update output directory label."""
        if self.output_dir.get():
            self.output_label.config(text=f"✓ Selected: {self.output_dir.get()}", fg="green")
        else:
            self.output_label.config(text="Default: ./output", fg="gray")
    
    def _log(self, message):
        """Add message to progress log."""
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update()
    
    def _validate_inputs(self):
        """Validate form entries."""
        if not self.input_dir.get():
            messagebox.showerror("Error", "Please select an input directory")
            return False
        
        if not Path(self.input_dir.get()).exists():
            messagebox.showerror("Error", f"Input directory not found:\n{self.input_dir.get()}")
            return False
        
        return True
    
    def _start_ingestion(self):
        """Start the ingestion process in a separate thread."""
        if not self._validate_inputs():
            return
        
        if self.is_processing:
            messagebox.showwarning("In Progress", "Ingestion already in progress")
            return
        
        self.is_processing = True
        self.ingest_button.config(state=tk.DISABLED)
        self.progress_text.delete("1.0", tk.END)
        
        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self._run_ingestion)
        thread.daemon = True
        thread.start()
    
    def _run_ingestion(self):
        """Execute the ingestion process."""
        try:
            self._log("=" * 60)
            self._log("SOUND BANK INGESTION STARTING")
            self._log("=" * 60)
            self._log("")
            
            self._log(f"Input Directory:   {self.input_dir.get()}")
            self._log(f"Output Directory:  {self.output_dir.get()}")
            self._log(f"Category:          {self.category.get()}")
            self._log(f"Normalization:     {self.normalize_mode.get()}")
            self._log(f"Target RMS:        {self.target_rms.get()}")
            self._log(f"Spectral Notch:    {'Yes (-12dB @ 1-3kHz)' if self.apply_notch.get() else 'No'}")
            self._log(f"Recursive:         {'Yes' if self.recursive.get() else 'No'}")
            self._log("")
            self._log("Starting ingestion...")
            self._log("")
            
            # Create engine
            engine = IngestionEngine(
                output_dir=self.output_dir.get(),
                target_rms=self.target_rms.get(),
                notch_attenuation_db=self.notch_db.get()
            )
            
            # Monkey-patch print to capture output
            import io
            import sys
            
            # Process directory
            processed, failed = engine.process_directory(
                input_dir=self.input_dir.get(),
                category=self.category.get(),
                apply_notch=self.apply_notch.get(),
                normalize_mode=self.normalize_mode.get(),
                recursive=self.recursive.get()
            )
            
            self._log("")
            self._log("=" * 60)
            self._log("INGESTION COMPLETE")
            self._log("=" * 60)
            self._log(f"Processed:  {processed} files")
            self._log(f"Failed:     {failed} files")
            self._log(f"Output:     {self.output_dir.get()}/master_bank.wav")
            self._log(f"Index:      {self.output_dir.get()}/bank.db")
            self._log("")
            
            if failed == 0:
                self._log("✓ All files processed successfully!")
                messagebox.showinfo("Success", f"Ingestion complete!\n\n{processed} files processed\n\nFiles saved to:\n{self.output_dir.get()}")
            else:
                self._log(f"⚠ {failed} files failed to process")
                messagebox.showwarning("Partial Success", f"Ingestion complete with {failed} errors\n\n{processed} files processed successfully")
        
        except Exception as e:
            self._log("")
            self._log(f"❌ ERROR: {str(e)}")
            self._log("")
            messagebox.showerror("Error", f"Ingestion failed:\n\n{str(e)}")
        
        finally:
            self.is_processing = False
            self.ingest_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = SoundBankIngestionGUI(root)
    root.mainloop()
