import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
import sys
import subprocess
from typing import Optional

class MasterDuelExporterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Master Duel Collection Exporter")
        self.root.geometry("500x750")  # Increased height to accommodate checkboxes
        self.root.minsize(500, 750)  # Increased minimum height
        
        # Track if a scan is in progress
        self.scan_in_progress = False
        
        # Initialize checkbox variables
        self.debug_mode = None
        self.print_summary = None
        
        # Configure terminal text tags for colors
        self.terminal = None  # Will be initialized in create_new_collection
        
        # Set application icon if available
        try:
            icon_path = Path(__file__).parent / "icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
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
        header_frame.pack(fill='x', pady=(0, 20))

        # Title centered in the frame
        title = ttk.Label(
            header_frame,
            text="Create New Collection",
            style='Title.TLabel',
            anchor='center'
        )
        title.pack(expand=True, fill='x')

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
            style='Subtitle.TLabel'
        )
        info.pack(pady=10)
        
        # Save location frame
        save_frame = ttk.Frame(self.main_frame)
        save_frame.pack(fill='x', padx=50, pady=15)
        
        # Save location label
        ttk.Label(save_frame, text="Select save folder:").pack(anchor='w')
        
        # Save location entry and browse button
        entry_frame = ttk.Frame(save_frame)
        entry_frame.pack(fill='x', pady=(5, 0))
        
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
            padding=10
        )
        instructions_frame.pack(fill='x', padx=20, pady=(20, 10))
        
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
        options_frame.pack(fill='x', padx=50, pady=(20, 10))

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
        button_frame.pack(fill='x', padx=50, pady=10)
        
        # Store buttons as instance variables for state management
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Collection Scan",
            command=self.start_collection_scan,
            style='TButton',
            padding=10
        )
        self.start_btn.pack(side='left', expand=True, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop Current Scan",
            command=self.stop_collection_scan,
            style='TButton',
            padding=10,
            state='disabled'  # Disabled by default
        )
        self.stop_btn.pack(side='right', expand=True, padx=5)
        
        # Terminal display frame
        terminal_frame = ttk.LabelFrame(
            self.main_frame,
            text="EXECUTION LOG",
            padding=5
        )
        terminal_frame.pack(fill='both', expand=True, padx=20, pady=(10, 20))
        
        # Create text widget for terminal output
        self.terminal = tk.Text(
            terminal_frame,
            bg='black',
            fg='#00ff00',  # Green text
            font=('Consolas', 10),
            wrap=tk.WORD,
            height=10,
            state='disabled',
            padx=5,
            pady=5
        )
        
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
        header_frame.pack(fill='x', pady=(0, 20))

        # Title centered in the frame
        title = ttk.Label(
            header_frame,
            text="Load a Collection",
            style='Title.TLabel',
            anchor='center'
        )
        title.pack(expand=True, fill='x')

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
            text="Select an existing collection CSV file to view or edit:",
            style='Subtitle.TLabel'
        )
        info.pack(pady=10)
        
        # File selection
        file_frame = ttk.Frame(self.main_frame)
        file_frame.pack(pady=20, fill='x', padx=50)
        
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
        
        # Load button
        load_btn = ttk.Button(
            self.main_frame,
            text="Load Collection",
            command=lambda: self.load_collection_file(self.file_entry.get()),
            style='TButton'
        )
        load_btn.pack(pady=20)
        
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

    def start_collection_scan(self):
        """Start the collection scanning process"""
        try:
            save_path = self.save_entry.get().strip()
            if not save_path:
                messagebox.showerror("Error", "Please specify a save location")
                return
                
                # Ensure the path is a directory and exists
            if os.path.isfile(save_path):
                # If it's a file, use its parent directory
                save_path = os.path.dirname(save_path)
                
            # Ensure the directory exists
            try:
                os.makedirs(save_path, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create directory: {str(e)}")
                return
            
            self.update_status("Starting collection scan...")
            
            # For now, just show a message with the save path
            messagebox.showinfo(
                "Scan Started", 
                f"The collection scan has started.\n\n"
                f"The collection will be saved to:\n{save_path}\n\n"
                "Please keep the Master Duel window visible."
            )
            
            # Here you would call your existing main() function with the save path
            # For example:
            # from main_new_better import main
            # main(output_path=save_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.update_status("Error during scan")

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

    def stop_collection_scan(self):
        """Handle the stop collection scan button click"""
        if self.scan_in_progress:
            self.scan_in_progress = False
            self.log("Scan stopped by user", "warning")
            self.start_btn['state'] = 'normal'
            self.stop_btn['state'] = 'disabled'
            # Add any additional cleanup code here when scan is stopped
            
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

            # For now, just show a message with the save path
            messagebox.showinfo(
                "Scan Started",
                f"The collection scan has started.\n\n"
                f"The collection will be saved to:\n{save_path}\n\n"
                "Please keep the Master Duel window visible."
            )
            
            # Here you would normally start the actual scanning process
            # For now, we'll simulate it with after()
            # self.root.after(5000, self.simulate_scan_completion)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.scan_in_progress = False
            self.start_btn['state'] = 'normal'
            self.stop_btn['state'] = 'disabled'
        self.root.update_idletasks()

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
