import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
import sys
import subprocess
from typing import Optional
import threading
import re

def parse_ansi(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Parse ANSI escape sequences and return clean text with tag ranges."""
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    clean_parts = []
    tag_ranges = []
    pos = 0
    current_tag = None
    last_end = 0
    for match in ansi_re.finditer(text):
        start, end = match.span()
        # Text before this code
        if start > last_end:
            clean_parts.append(text[last_end:start])
            if current_tag:
                tag_ranges.append((pos, pos + (start - last_end), current_tag))
            pos += start - last_end
        # Update tag
        code = match.group()
        if code == '\x1b[0m':
            current_tag = None
        elif code == '\x1b[94m':
            current_tag = 'blue'
        elif code == '\x1b[97m':
            current_tag = 'white'
        elif code == '\x1b[92m':
            current_tag = 'green'
        elif code == '\x1b[93m':
            current_tag = 'yellow'
        elif code == '\x1b[91m':
            current_tag = 'red'
        elif code == '\x1b[96m':
            current_tag = 'cyan'
        elif code == '\x1b[1;97m':
            current_tag = 'bold_white'
        elif code == '\x1b[1;96m':
            current_tag = 'bold_cyan'
        elif code == '\x1b[1;93m':
            current_tag = 'bold_yellow'
        elif code == '\x1b[1;94m':
            current_tag = 'bold_blue'
        elif code == '\x1b[4;97m':
            current_tag = 'underline_white'
        last_end = end
    # Remaining text
    if last_end < len(text):
        clean_parts.append(text[last_end:])
        if current_tag:
            tag_ranges.append((pos, pos + (len(text) - last_end), current_tag))
    clean_text = ''.join(clean_parts)
    return clean_text, tag_ranges

class MasterDuelExporterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Master Duel Collection Exporter")
        self.root.geometry("500x750+0+0")  # Increased height to accommodate checkboxes
        self.root.minsize(500, 750)  # Increased minimum height
        
        # Track if a scan is in progress
        self.scan_in_progress = False

        # Initialize checkbox variables
        self.debug_mode = None
        self.print_summary = None

        # Process for scanning
        self.process = None
        
        # Configure terminal text tags for colors
        self.terminal = None  # Will be initialized in create_new_collection
        
        # Set application icon if available
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'favicon.ico')
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Could not load icon: {e}")

        # Configure style
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 12), padding=10)
        self.style.configure('Small.TButton', font=('Arial', 9), padding=5)
        self.style.configure('Title.TLabel', font=('Arial', 18, 'bold'), anchor='center')
        self.style.configure('Subtitle.TLabel', font=('Arial', 12))
        
        # Configure terminal text tags
        if hasattr(self, 'terminal') and self.terminal:
            self.terminal.tag_configure('timestamp', foreground='#808080')  # Gray
            self.terminal.tag_configure('info', foreground='#00ff00')  # Green
            self.terminal.tag_configure('warning', foreground='#ffff00')  # Yellow
            self.terminal.tag_configure('error', foreground='#ff0000')  # Red
            self.terminal.tag_configure('debug', foreground='#00ffff')  # Cyan
        
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Show the main menu initially
        self.show_main_menu()

    def clear_frame(self):
        """Clear all widgets from the main frame"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_main_menu(self):
        """Show the main menu with two main options"""
        self.clear_frame()
        
        # Title
        title = ttk.Label(
            self.main_frame,
            text="Master Duel Collection Exporter",
            style='Title.TLabel'
        )
        title.pack(pady=(20, 40))
        
        # Buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(expand=True)
        
        # Create New Collection Button
        new_btn = ttk.Button(
            button_frame,
            text="Create New Collection CSV",
            command=self.create_new_collection,
            width=25
        )
        new_btn.pack(pady=10)
        
        # Load a Collection Button
        load_btn = ttk.Button(
            button_frame,
            text="Load a Collection CSV",
            command=self.load_existing_collection,
            width=25
        )
        load_btn.pack(pady=10)
        
        # Help text
        help_text = ttk.Label(
            self.main_frame,
            text="Contact Skybullet07 on Discord for help, or to report issues.",
            style='Subtitle.TLabel',
            foreground='gray50'
        )
        help_text.pack(side=tk.BOTTOM, pady=10)

    def create_new_collection(self):
        """Handle Create New Collection button click"""
        self.clear_frame()
        
        # Default save location (directory only, no default filename)
        self.default_save_dir = os.path.join(os.getcwd(), 'collection_csv')
        os.makedirs(self.default_save_dir, exist_ok=True)
        self.save_path = self.default_save_dir  # Just the directory, no default filename
        
        # Header frame for back button and title
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill='x', pady=(0, 14))  # Reduced from 20 to 14 (30% less)

        # Title centered in the frame
        title = ttk.Label(
            header_frame,
            text="Create New Collection",
            style='Title.TLabel',
            anchor='center',
            padding=(0, 0, 0, 0)  # No padding here, controlled by pack
        )
        title.pack(expand=True, fill='x', pady=(0, 2))  # Reduced bottom padding to 2

        # Back button placed on the left
        back_btn = ttk.Button(
            header_frame,
            text="← Back",
            command=self.show_main_menu,
            style='Small.TButton'
        )
        back_btn.place(x=0, y=0, anchor='nw')
        
        info = ttk.Label(
            self.main_frame,
            text="Save your Master Duel collection to a new CSV file.",
            style='Subtitle.TLabel',
            padding=(0, 0, 0, 0)  # No padding here
        )
        info.pack(pady=(0, 7))  # Only bottom padding of 7
        
        # Save location frame with label frame
        save_selector_frame = ttk.LabelFrame(
            self.main_frame,
            text="Select save folder:",
            padding=4
        )
        save_selector_frame.pack(fill='x', padx=35, pady=7)  # Matches other pages
        
        # Save location entry and browse button
        entry_frame = ttk.Frame(save_selector_frame)
        entry_frame.pack(fill='x', padx=5, pady=5)
        
        self.save_entry = ttk.Entry(entry_frame)
        self.save_entry.insert(0, self.save_path)
        self.save_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(
            entry_frame,
            text="...",
            command=self.browse_save_location,
            style='Small.TButton',
            width=3
        )
        browse_btn.pack(side='right')
        
        # Instructions section
        instructions_frame = ttk.LabelFrame(
            self.main_frame,
            text="INSTRUCTIONS",
            padding=7  # Reduced from 10 to 7
        )
        instructions_frame.pack(fill='x', padx=14, pady=(14, 7))  # Reduced padding by 30%
        
        instructions_text = (
            "1. Open Master Duel and click on the \"Deck\" button. \n"
            "2. Click on any of your Decks, then click \"Edit Deck\" \n"
            "3. Click \"Start Collection Scan\". \n"
            "4. DON'T close/minimise Master Duel or move your cursor during scan.\n"
            "5. Click \"Stop Current Scan\" to stop scanning early.\n"
            "6. Results will be saved to a CSV file in the selected folder."
        )
        
        ttk.Label(
            instructions_frame,
            text=instructions_text,
            justify='left',
            padding=(5, 5, 5, 5)
        ).pack(anchor='w')
        
        # Options frame for checkboxes
        options_frame = ttk.Frame(self.main_frame)
        options_frame.pack(fill='x', padx=35, pady=(14, 7))  # Reduced padding by 30%

        # Configure grid for centering
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=0)
        options_frame.columnconfigure(2, weight=0)
        options_frame.columnconfigure(3, weight=1)

        # Debug Mode checkbox
        self.debug_mode = tk.BooleanVar(value=False)
        debug_check = ttk.Checkbutton(
            options_frame,
            text="Debug Mode",
            variable=self.debug_mode,
            command=self.log_debug_mode
        )
        debug_check.grid(row=0, column=1, padx=20)

        # Print Summary checkbox
        self.print_summary = tk.BooleanVar(value=False)
        summary_check = ttk.Checkbutton(
            options_frame,
            text="Print Summary",
            variable=self.print_summary,
            command=self.log_print_summary
        )
        summary_check.grid(row=0, column=2, padx=20)
        
        # Button frame for Start/Stop
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill='x', padx=35, pady=7)  # Reduced padding by 30%
        
        # Store buttons as instance variables for state management
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Collection Scan",
            command=self.start_collection_scan,
            style='Small.TButton',
            padding=5
        )
        self.start_btn.pack(side='left', expand=True, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop Current Scan",
            command=self.stop_collection_scan,
            style='Small.TButton',
            padding=5,
            state='disabled'  # Disabled by default
        )
        self.stop_btn.pack(side='right', expand=True, padx=5)
        
        # Terminal display frame
        terminal_frame = ttk.LabelFrame(
            self.main_frame,
            text="EXECUTION LOG",
            padding=4  # Reduced from 5 to 4
        )
        terminal_frame.pack(fill='both', expand=True, padx=14, pady=(7, 14))  # Reduced padding by 30%
        
        # Create text widget for terminal output
        self.terminal = tk.Text(
            terminal_frame,
            bg='black',
            fg='white',  # White text
            font=('Consolas', 10),
            wrap=tk.WORD,
            height=10,
            state='disabled',
            padx=5,
            pady=5
        )

        # Configure ANSI color tags
        self.terminal.tag_configure('blue', foreground='#0080ff')
        self.terminal.tag_configure('white', foreground='white')
        self.terminal.tag_configure('green', foreground='#00ff00')
        self.terminal.tag_configure('yellow', foreground='#ffff00')
        self.terminal.tag_configure('red', foreground='#ff0000')
        self.terminal.tag_configure('cyan', foreground='#00ffff')
        self.terminal.tag_configure('bold_white', foreground='white', font=('Consolas', 10, 'bold'))
        self.terminal.tag_configure('bold_cyan', foreground='#00ffff', font=('Consolas', 10, 'bold'))
        self.terminal.tag_configure('bold_yellow', foreground='#ffff00', font=('Consolas', 10, 'bold'))
        self.terminal.tag_configure('bold_blue', foreground='#0080ff', font=('Consolas', 10, 'bold'))
        self.terminal.tag_configure('underline_white', foreground='white', underline=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(terminal_frame, orient='vertical', command=self.terminal.yview)
        self.terminal.configure(yscrollcommand=scrollbar.set)

        # Pack the scrollbar and terminal
        scrollbar.pack(side='right', fill='y')
        self.terminal.pack(side='left', fill='both', expand=True)
        
        # Add initial message
        self.log("Terminal initialised.", "info")
        self.update_status("Ready to start a new export.")

    def load_existing_collection(self):
        """Handle Load a Collection button click"""
        self.clear_frame()
        
        # Header frame for back button and title
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill='x', pady=(0, 14))  # Matches create page

        # Title centered in the frame
        title = ttk.Label(
            header_frame,
            text="Load a Collection",
            style='Title.TLabel',
            anchor='center'
        )
        title.pack(expand=True, fill='x', pady=(0, 2))  # Matches create page

        # Back button placed on the left
        back_btn = ttk.Button(
            header_frame,
            text="← Back",
            command=self.show_main_menu,
            style='Small.TButton'
        )
        back_btn.place(x=0, y=0, anchor='nw')
        
        # Info text with reduced bottom padding
        info = ttk.Label(
            self.main_frame,
            text="Select an existing collection CSV file to view or edit:",
            style='Subtitle.TLabel'
        )
        info.pack(pady=(0, 7))  # Reduced from 10 to 7 to match create page
        
        # File selection frame with consistent padding
        file_selector_frame = ttk.LabelFrame(
            self.main_frame,
            text="Select saved collection:",
            padding=4
        )
        file_selector_frame.pack(fill='x', padx=35, pady=7)  # Matches create page
        
        # File entry and browse button
        file_frame = ttk.Frame(file_selector_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        self.file_entry = ttk.Entry(file_frame)
        self.file_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(
            file_frame,
            text="...",
            command=self.browse_file,
            style='Small.TButton',
            width=3
        )
        browse_btn.pack(side='right')
        
        # Load button with consistent styling
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill='x', padx=35, pady=7)  # Matches create page
        
        load_btn = ttk.Button(
            button_frame,
            text="Load Collection",
            command=lambda: self.load_collection_file(self.file_entry.get()),
            style='Small.TButton',
            padding=5
        )
        load_btn.pack(expand=True)
        
        # Status message at the bottom
        self.update_status("Select a collection file to load")

    def browse_file(self):
        """Open file dialog to select a CSV file"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select Collection File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            
    def browse_save_location(self):
        """Open directory dialog to choose where to save the CSV"""
        from tkinter import filedialog
        initial_dir = self.save_entry.get() or self.default_save_dir
        
        # Ask for directory instead of filename
        directory = filedialog.askdirectory(
            title="Select Save Folder",
            mustexist=False,  # Allow creating new directories
            initialdir=initial_dir
        )
        
        if directory:  # If user didn't cancel
            self.save_entry.delete(0, tk.END)
            self.save_entry.insert(0, directory)



    def load_collection_file(self, filepath):
        """Load an existing collection file"""
        if not filepath:
            messagebox.showerror("Error", "Please select a file first")
            return
            
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File not found: {filepath}")
            return
            
        try:
            # TODO: Implement loading and displaying the collection
            self.update_status(f"Loaded collection: {os.path.basename(filepath)}")
            messagebox.showinfo(
                "Success", 
                f"Successfully loaded collection from:\n{filepath}"
            )
            # Here you would add code to display the loaded collection
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load collection: {str(e)}")
            self.update_status("Error loading collection")

    def log(self, message: str, level: str = "info"):
        """Add a message to the terminal with specified log level

        Args:
            message: The message to display
            level: Log level ('info', 'warning', 'error')
        """
        if self.terminal is None:
            print(f"[{level.upper()}] {message}")
            return

        # Map log levels to colors
        colors = {
            'info': '#00ff00',  # Green
            'warning': '#ffff00',  # Yellow
            'error': '#ff0000',  # Red
            'debug': '#00ffff'   # Cyan
        }

        # Get current time
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Format the message with timestamp and log level
        log_message = f"[{timestamp}] {message}\n"

        # Enable text widget for editing
        self.terminal.configure(state='normal')

        # Insert the message with appropriate color
        self.terminal.insert('end', f"[{timestamp}] ", 'timestamp')
        self.terminal.insert('end', f"{message}\n", level)

        # Auto-scroll to bottom
        self.terminal.see('end')

        # Disable text widget to prevent user editing
        self.terminal.configure(state='disabled')

        # Also print to console for debugging
        print(f"[{level.upper()}] {log_message}", end='')
    
    def update_status(self, message: str):
        """Update the status message and log it"""
        self.log(f"Status: {message}", "info")

    def log_debug_mode(self):
        """Log debug mode toggle"""
        if self.debug_mode.get():
            self.log("Debug mode ENABLED", "info")
        else:
            self.log("Debug mode DISABLED", "info")

    def log_print_summary(self):
        """Log print summary toggle"""
        if self.print_summary.get():
            self.log("Print summary ENABLED", "info")
        else:
            self.log("Print summary DISABLED", "info")


            
    def start_collection_scan(self):
        """Start the collection scanning process"""
        try:
            save_path = self.save_entry.get().strip()
            if not save_path:
                messagebox.showerror("Error", "Please specify a save location")
                self.log("Error: No save location specified", "error")
                return

            # Ensure the path is a directory and exists
            if os.path.isfile(save_path):
                save_path = os.path.dirname(save_path)
                self.log(f"Using parent directory: {save_path}", "debug")

            # Ensure the directory exists
            try:
                os.makedirs(save_path, exist_ok=True)
                self.log(f"Using save directory: {save_path}", "info")
            except Exception as e:
                error_msg = f"Could not create directory: {str(e)}"
                messagebox.showerror("Error", error_msg)
                self.log(error_msg, "error")
                return

            # Update UI state
            self.scan_in_progress = True
            self.start_btn['state'] = 'disabled'
            self.stop_btn['state'] = 'normal'

            self.log("Initializing collection scan...", "info")

            # Prepare command
            cmd = [sys.executable, '-u', 'main-new-better.py']
            if self.debug_mode.get():
                cmd.append('--debug')
            if not self.print_summary.get():
                cmd.append('--no-summary')
            cmd.extend(['--output-dir', save_path])

            # Start subprocess
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.getcwd())

            # Start thread to read output
            threading.Thread(target=self.read_output, daemon=True).start()

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.scan_in_progress = False
            self.start_btn['state'] = 'normal'
            self.stop_btn['state'] = 'disabled'
        self.root.update_idletasks()

    def read_output(self):
        """Read output from subprocess and log to terminal"""
        from datetime import datetime
        summary_mode = False
        for line in iter(self.process.stdout.readline, ''):
            clean_line, ranges = parse_ansi(line)
            stripped = clean_line.strip()
            if stripped == "=== FINAL CARD SUMMARY ===":
                summary_mode = True
            elif "Preparing CSV file data" in stripped:
                summary_mode = False
            elif stripped == "=== Process Complete ===":
                summary_mode = False
            if summary_mode:
                full_line = clean_line
                ts_end = 0
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                ts_prefix = f"[{timestamp}] "
                full_line = ts_prefix + clean_line
                ts_end = len(ts_prefix)
            self.terminal.configure(state='normal')
            start_idx = self.terminal.index('end-1c')
            self.terminal.insert('end', full_line)
            if not summary_mode:
                # Apply timestamp tag
                self.terminal.tag_add('timestamp', f"{start_idx}+0c", f"{start_idx}+{ts_end}c")
            # Apply ANSI tags, adjusted for prefix
            for s, e, tag in ranges:
                adj_s = s + ts_end
                adj_e = e + ts_end
                tag_start = f"{start_idx}+{adj_s}c"
                tag_end = f"{start_idx}+{adj_e}c"
                self.terminal.tag_add(tag, tag_start, tag_end)
            self.terminal.see('end')
            self.terminal.configure(state='disabled')
        self.process.stdout.close()
        self.process.wait()
        # Reset UI
        self.scan_in_progress = False
        self.start_btn['state'] = 'normal'
        self.stop_btn['state'] = 'disabled'
        self.log("Execution completed.")

    def stop_collection_scan(self):
        """Handle the stop collection scan button click"""
        if self.scan_in_progress and self.process:
            self.process.terminate()
            self.log("Scan stopped by user.", "warning")
        self.scan_in_progress = False
        self.start_btn['state'] = 'normal'
        self.stop_btn['state'] = 'disabled'

def main():
    root = tk.Tk()
    app = MasterDuelExporterApp(root)
    root.mainloop()

if __name__ == "__main__":
    """
    To change the application icon:
    1. Place your icon file (must be .ico format on Windows) in the same directory as this script
    2. Name it 'icon.ico' or update the icon_path in the __init__ method
    3. The icon will be automatically loaded when the application starts
    
    For best results, use an .ico file with multiple sizes (16x16, 32x32, 48x48, etc.)
    """
    main()
