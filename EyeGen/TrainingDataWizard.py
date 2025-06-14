import tkinter as tk
from tkinter import messagebox, scrolledtext
import csv
import numpy as np

class TrainingDataGeneratorUI:
    """
    A GUI application to generate and edit training_data.csv with shape preview and drawing capability.
    Users can:
    - Input 8 numerical inputs and 16 (x,y) coordinate pairs (32 total outputs).
    - Add new data rows.
    - Preview the shape formed by the coordinates.
    - Draw a shape directly on the canvas and use its coordinates to populate the output fields.
    - Load an existing row from the displayed data for editing.
    - Update a loaded row with new values from the input/output fields.
    - Delete a selected row.
    - Cycle through all loaded data rows with smooth interpolation animations.
    - Save all accumulated data to a CSV file.
    - Loads existing data from 'training_data.csv' on startup.
    - Displays the orientation (clockwise/anti-clockwise) of the current shape.
    - Flip the orientation of the current shape by reversing its coordinate order.
    """
    def __init__(self, master):
        """
        Initializes the GUI application.

        Args:
            master (tk.Tk): The root Tkinter window.
        """
        self.master = master
        master.title("Training Data CSV Generator with Drawing and Editing")
        master.geometry("1000x750") # Set initial window size
        master.resizable(True, True) # Allow window resizing

        self.data_rows = [] # To store all accumulated data rows
        self.current_drawing_points = [] # Stores pixel coordinates of the current drawn path
        self.drawing_active = False # Flag to indicate if mouse button is pressed for drawing
        self.editing_row_index = -1 # Stores the 0-based index of the row currently loaded for editing
        self.cycle_index = -1 # Stores the index of the row currently displayed/interpolated to

        # Interpolation parameters
        self.interpolation_steps = 25 # Number of frames for interpolation
        self.interpolation_delay_ms = 40 # Delay between frames in milliseconds
        self.interpolating = False # Flag to indicate if an interpolation is active
        self.interpolation_job = None # Stores the after() job ID for cancellation
        self.start_interpolation_coords_flat = None # Start coordinates for interpolation (flattened)
        self.end_interpolation_coords_flat = None # End coordinates for interpolation (flattened)
        self.current_interpolation_step = 0 # Current step in the interpolation animation

        # --- Configure Grid Layout ---
        master.grid_columnconfigure(0, weight=1) # Left Panel
        master.grid_columnconfigure(1, weight=1) # Right Panel (Canvas)

        master.grid_rowconfigure(0, weight=0) # Input Frame
        master.grid_rowconfigure(1, weight=0) # Output Frame
        master.grid_rowconfigure(2, weight=0) # Button Frame
        master.grid_rowconfigure(3, weight=0) # Status Label
        master.grid_rowconfigure(4, weight=1) # Data Display (expandable)

        # --- Left Panel: Input, Output, Buttons, Data Display ---
        left_panel = tk.Frame(master)
        left_panel.grid(row=0, column=0, rowspan=10, padx=10, pady=5, sticky="nsew")
        left_panel.grid_columnconfigure(0, weight=1)

        # --- Input Section (8 numerical inputs) ---
        input_frame = tk.LabelFrame(left_panel, text="8 Input Values", padx=10, pady=10)
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.input_entries = []
        for i in range(8):
            label = tk.Label(input_frame, text=f"Input {i+1}:")
            label.grid(row=i // 4, column=(i % 4) * 2, padx=5, pady=2, sticky="e")
            entry = tk.Entry(input_frame, width=10)
            entry.grid(row=i // 4, column=(i % 4) * 2 + 1, padx=5, pady=2, sticky="w")
            self.input_entries.append(entry)
            input_frame.grid_columnconfigure((i % 4) * 2, weight=1)
            input_frame.grid_columnconfigure((i % 4) * 2 + 1, weight=1)

        # --- Output Section (16 (x,y) coordinate pairs = 32 outputs) ---
        output_frame = tk.LabelFrame(left_panel, text="16 (x,y) Output Coordinates", padx=10, pady=10)
        output_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.output_entries = [] # Stores [x1, y1, x2, y2, ...] entries
        for i in range(16): # 16 pairs
            label = tk.Label(output_frame, text=f"Coord {i+1}:")
            label.grid(row=i // 4, column=(i % 4) * 4, padx=5, pady=2, sticky="e")
            entry_x = tk.Entry(output_frame, width=7)
            entry_x.grid(row=i // 4, column=(i % 4) * 4 + 1, padx=2, pady=2, sticky="ew")
            label_comma = tk.Label(output_frame, text=",")
            label_comma.grid(row=i // 4, column=(i % 4) * 4 + 2, padx=0, pady=2, sticky="w")
            entry_y = tk.Entry(output_frame, width=7)
            entry_y.grid(row=i // 4, column=(i % 4) * 4 + 3, padx=2, pady=2, sticky="ew")
            self.output_entries.extend([entry_x, entry_y])
            output_frame.grid_columnconfigure((i % 4) * 4, weight=0)
            output_frame.grid_columnconfigure((i % 4) * 4 + 1, weight=1)
            output_frame.grid_columnconfigure((i % 4) * 4 + 2, weight=0)
            output_frame.grid_columnconfigure((i % 4) * 4 + 3, weight=1)

        # --- Buttons ---
        button_frame = tk.Frame(left_panel)
        button_frame.grid(row=2, column=0, padx=5, pady=10)

        self.add_button = tk.Button(button_frame, text="Add Data Row", command=self.add_row)
        self.add_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear Fields", command=self.clear_fields)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.preview_button = tk.Button(button_frame, text="Preview Shape (from fields)", command=self.preview_shape_from_fields)
        self.preview_button.pack(side=tk.LEFT, padx=5)

        self.use_drawn_shape_button = tk.Button(button_frame, text="Use Drawn Shape for Output", command=self.use_drawn_shape_for_output)
        self.use_drawn_shape_button.pack(side=tk.LEFT, padx=5)

        # Buttons for editing/deleting
        self.load_edit_button = tk.Button(button_frame, text="Load Selected Row", command=self.load_selected_row)
        self.load_edit_button.pack(side=tk.LEFT, padx=5)

        self.update_button = tk.Button(button_frame, text="Update Loaded Row", command=self.update_loaded_row)
        self.update_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_button = tk.Button(button_frame, text="Delete Selected Row", command=self.delete_selected_row)
        self.delete_button.pack(side=tk.LEFT, padx=5)

        # Button for flipping orientation - NEW
        self.flip_orientation_button = tk.Button(button_frame, text="Flip Orientation", command=self.flip_shape_orientation)
        self.flip_orientation_button.pack(side=tk.LEFT, padx=5)

        # Button for cycling with interpolation
        self.cycle_button = tk.Button(button_frame, text="Cycle Through Shapes", command=self._start_cycle_interpolation)
        self.cycle_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(button_frame, text="Save to training_data.csv", command=self.save_to_csv)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.exit_button = tk.Button(button_frame, text="Exit", command=master.quit)
        self.exit_button.pack(side=tk.LEFT, padx=5)


        
        # Add the new button here, after other buttons are packed
        self.normalize_scale_button = tk.Button(button_frame, text="Normalize Shape Scale", command=self._normalize_shape_scale)
        self.normalize_scale_button.pack(side=tk.LEFT, padx=5)

        self.normalize_scale_button = tk.Button(master, text="Rotate Points", command=self._rotate_points)
        self.normalize_scale_button.grid(row=0,column = 0,sticky = 'es')

        # --- Status and Current Data Display ---
        self.status_label = tk.Label(left_panel, text="Current Data Rows (0): Click on a row number to select for editing/deletion.", anchor="w")
        self.status_label.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        # Make data_display not directly editable by user, only programmatically updated.
        # User will select by placing cursor.
        self.data_display = scrolledtext.ScrolledText(left_panel, height=10, state='disabled', wrap=tk.WORD)
        self.data_display.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

        # --- Right Panel: Canvas for Shape Visualization and Drawing ---
        canvas_frame = tk.LabelFrame(master, text="Draw Your Shape Here (Click and Drag)", padx=10, pady=10)
        canvas_frame.grid(row=0, column=1, rowspan=5, padx=10, pady=5, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(1, weight=0) # Row for orientation label
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height,
                                bg="lightgray", borderwidth=2, relief="sunken") # Changed background for drawing clarity
        self.canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Orientation Label - New
        self.orientation_label = tk.Label(canvas_frame, text="Orientation: N/A", anchor="center", font=("TkDefaultFont", 10, "bold"))
        self.orientation_label.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self._start_drawing)
        self.canvas.bind("<B1-Motion>", self._draw_line)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drawing)

        # Initial clear of canvas
        self.canvas.delete("all")
        self._update_orientation_label("N/A") # Initialize label text

        # --- Load existing data on startup ---
        self._load_data_from_csv('training_data.csv')

    def _rotate_points(self):
        """
        Scales the shape currently defined in the output fields so that its
        furthest point from the centroid is normalized to a target value (e.g., 50.0),
        and then updates the output fields.
        """
        self._cancel_interpolation() # Stop any active interpolation
        output_values = self._validate_and_get_values(self.output_entries)
        if output_values is None:
            messagebox.showerror("Scaling Error", "All output (coordinate) fields must contain valid numerical values for scaling.")
            return
        if len(output_values) != 32:
            messagebox.showerror("Scaling Error", "Expected 32 output values (16 pairs) to scale the shape.")
            return

        # Convert flattened list to (x,y) tuples
        coordinates = []
        for i in range(0, len(output_values), 2):
            coordinates.append((output_values[i], output_values[i+1]))

        if not coordinates:
            messagebox.showwarning("Scaling Info", "No coordinates to scale.")
            return

        # Target max distance from center is 50.0, assuming a 0-100 coordinate system where center is (50,50)
        print(coordinates)
        rotated_coords_tuples = coordinates.copy()
        rotated_coords_tuples.append(rotated_coords_tuples.pop(0))

        # Flatten back to a list of 32 values
        scaled_output_values = []
        for x, y in rotated_coords_tuples:
            scaled_output_values.append(x)
            scaled_output_values.append(y)

        # Populate the output entries with the new, scaled values
        for i, val in enumerate(scaled_output_values):
            if i < len(self.output_entries):
                self.output_entries[i].delete(0, tk.END)
                self.output_entries[i].insert(0, f"{val:.4f}")

        messagebox.showinfo("Shape Scaled", "Shape has been scaled and centered based on its furthest point from the centroid.")
        
        self.preview_shape_from_fields() # Redraw with new scale and update orientation label

        #self.update_loaded_row()
        #self.load_selected_row()





    # Add this method inside the TrainingDataGeneratorUI class
    def _calculate_centroid_and_furthest_distance(self, coordinates):
        """
        Calculates the centroid and the maximum distance of any point from the centroid.

        Args:
            coordinates (list of tuple): A list of (x, y) tuples.

        Returns:
            tuple: (centroid_x, centroid_y, max_distance)
                   Returns (None, None, 0.0) if coordinates are empty.
        """
        if not coordinates:
            return None, None, 0.0

        coords_array = np.array(coordinates)
        centroid_x = np.mean(coords_array[:, 0])
        centroid_y = np.mean(coords_array[:, 1])

        distances = np.sqrt(np.sum((coords_array - np.array([centroid_x, centroid_y]))**2, axis=1))
        max_distance = np.max(distances) if distances.size > 0 else 0.0

        return centroid_x, centroid_y, max_distance

    # Add this method inside the TrainingDataGeneratorUI class
    def _scale_shape_to_max_distance_from_center(self, coordinates, target_max_distance=50.0):
        """
        Scales a list of coordinates such that the furthest point from their
        centroid is equal to `target_max_distance`.
        The shape is also translated so its centroid is at (target_max_distance, target_max_distance)
        to effectively center it within a 0-100 coordinate system, assuming a target_max_distance of 50.

        Args:
            coordinates (list of tuple): A list of (x, y) tuples.
            target_max_distance (float): The desired maximum distance from the center.
                                         For 0-100 range, 50 is a good target for max extent from center.

        Returns:
            list: A new list of (x, y) tuples with scaled and centered coordinates.
                  Returns empty list if input coordinates are empty.
        """
        if not coordinates:
            return []

        centroid_x, centroid_y, current_max_distance = self._calculate_centroid_and_furthest_distance(coordinates)

        if current_max_distance == 0: # All points are the same or only one point
            return coordinates # Cannot scale, return original

        scale_factor = target_max_distance / current_max_distance

        scaled_coordinates = []
        new_center_x = target_max_distance # For a 0-100 range, (50,50) is the center
        new_center_y = target_max_distance

        for x, y in coordinates:
            # Translate to origin, scale, then translate to new desired center
            scaled_x = (x - centroid_x) * scale_factor + new_center_x
            scaled_y = (y - centroid_y) * scale_factor + new_center_y
            scaled_coordinates.append((scaled_x, scaled_y))

        return scaled_coordinates

    # Add this method inside the TrainingDataGeneratorUI class
    def _normalize_shape_scale(self):
        """
        Scales the shape currently defined in the output fields so that its
        furthest point from the centroid is normalized to a target value (e.g., 50.0),
        and then updates the output fields.
        """
        self._cancel_interpolation() # Stop any active interpolation
        output_values = self._validate_and_get_values(self.output_entries)
        if output_values is None:
            messagebox.showerror("Scaling Error", "All output (coordinate) fields must contain valid numerical values for scaling.")
            return
        if len(output_values) != 32:
            messagebox.showerror("Scaling Error", "Expected 32 output values (16 pairs) to scale the shape.")
            return

        # Convert flattened list to (x,y) tuples
        coordinates = []
        for i in range(0, len(output_values), 2):
            coordinates.append((output_values[i], output_values[i+1]))

        if not coordinates:
            messagebox.showwarning("Scaling Info", "No coordinates to scale.")
            return

        # Target max distance from center is 50.0, assuming a 0-100 coordinate system where center is (50,50)
        scaled_coords_tuples = self._scale_shape_to_max_distance_from_center(coordinates, target_max_distance=50.0)

        # Flatten back to a list of 32 values
        scaled_output_values = []
        for x, y in scaled_coords_tuples:
            scaled_output_values.append(x)
            scaled_output_values.append(y)

        # Populate the output entries with the new, scaled values
        for i, val in enumerate(scaled_output_values):
            if i < len(self.output_entries):
                self.output_entries[i].delete(0, tk.END)
                self.output_entries[i].insert(0, f"{val:.4f}")

        messagebox.showinfo("Shape Scaled", "Shape has been scaled and centered based on its furthest point from the centroid.")
        self.preview_shape_from_fields() # Redraw with new scale and update orientation label


    def _validate_and_get_values(self, entries):
        """
        Validates if all entries contain numerical values and returns them as a list of floats.

        Args:
            entries (list): A list of Tkinter Entry widgets.

        Returns:
            list: A list of float values if all are valid, None otherwise.
        """
        values = []
        for entry in entries:
            try:
                value = float(entry.get())
                values.append(value)
            except ValueError:
                return None # Indicate error
        return values

    def _populate_fields_from_row_data(self, row_data):
        """Helper to populate input and output entries from a given row_data list."""
        for entry in self.input_entries:
            entry.delete(0, tk.END)
        for i, val in enumerate(row_data[:8]):
            self.input_entries[i].insert(0, f"{val:.4f}")

        for entry in self.output_entries:
            entry.delete(0, tk.END)
        for i, val in enumerate(row_data[8:]):
            self.output_entries[i].insert(0, f"{val:.4f}")

    def add_row(self):
        """
        Collects input and output values, validates them, and adds a new row
        to the internal data_rows list. Updates the display and status.
        """
        self._cancel_interpolation() # Stop any active interpolation
        input_values = self._validate_and_get_values(self.input_entries)
        if input_values is None:
            messagebox.showerror("Input Error", "All input fields must contain valid numerical values.")
            return

        output_values = self._validate_and_get_values(self.output_entries)
        if output_values is None:
            messagebox.showerror("Input Error", "All output (coordinate) fields must contain valid numerical values.")
            return

        if len(input_values) != 8:
            messagebox.showerror("Validation Error", "Please provide exactly 8 input values.")
            return

        if len(output_values) != 32:
            messagebox.showerror("Validation Error", "Please provide exactly 32 output values (16 pairs).")
            return

        new_row = input_values + output_values
        self.data_rows.append(new_row)
        self.update_display()
        self.clear_fields() # Clear fields after adding a row
        messagebox.showinfo("Success", "Data row added successfully!")

    def clear_fields(self):
        """
        Clears all input and output entry fields and the current drawn shape.
        Resets the editing state and stops any active interpolation.
        """
        self._cancel_interpolation() # Stop any active interpolation
        for entry in self.input_entries:
            entry.delete(0, tk.END)
        for entry in self.output_entries:
            entry.delete(0, tk.END)
        self.current_drawing_points = []
        self.clear_canvas() # Also clear canvas when clearing fields
        self.editing_row_index = -1 # Reset editing state
        self.cycle_index = -1 # Reset cycle index
        self._update_orientation_label("N/A") # Clear orientation label

    def clear_fields_without_resetting_editing_index(self):
        """Helper to clear fields without affecting self.editing_row_index or self.cycle_index."""
        for entry in self.input_entries:
            entry.delete(0, tk.END)
        for entry in self.output_entries:
            entry.delete(0, tk.END)
        self.current_drawing_points = []
        self.clear_canvas()
        self._update_orientation_label("N/A")

    def clear_canvas(self):
        """Clears all drawings from the canvas."""
        self.canvas.delete("all")
        self._update_orientation_label("N/A") # Clear orientation when canvas is cleared

    def _start_drawing(self, event):
        """Starts a new drawing session on the canvas."""
        self._cancel_interpolation() # Stop any active interpolation
        self.clear_canvas() # Clear previous drawing
        self.current_drawing_points = []
        self.drawing_active = True
        self.current_drawing_points.append((event.x, event.y))
        # Draw the first point
        self.canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="darkblue", outline="darkblue")
        self._update_orientation_label("Drawing...") # Update while drawing

    def _draw_line(self, event):
        """Draws lines as the mouse is dragged."""
        if self.drawing_active:
            x, y = event.x, event.y
            if not self.current_drawing_points: # Should not happen if _start_drawing works
                self.current_drawing_points.append((x, y))
                return

            last_x, last_y = self.current_drawing_points[-1]
            # Draw line segment
            self.canvas.create_line(last_x, last_y, x, y, fill="darkblue", width=2)
            self.current_drawing_points.append((x, y))

    def _stop_drawing(self, event):
        """Stops the drawing session."""
        self.drawing_active = False
        if len(self.current_drawing_points) > 1:
            # Draw a line from the last point to the first to close the shape visually
            first_x, first_y = self.current_drawing_points[0]
            last_x, last_y = self.current_drawing_points[-1]
            self.canvas.create_line(last_x, last_y, first_x, first_y, fill="darkblue", width=2, dash=(4, 2)) # Dashed line to show closure
            self._update_orientation_label("Ready to use drawn shape.")

    def use_drawn_shape_for_output(self):
        """
        Processes the currently drawn shape, resamples it to 16 points,
        normalizes the coordinates, and populates the output entry fields.
        """
        self._cancel_interpolation() # Stop any active interpolation
        if len(self.current_drawing_points) < 16:
            messagebox.showwarning("Drawing Error", "Please draw at least 16 points to define the shape.")
            self._update_orientation_label("Drawing too short (needs >= 16 pts)")
            return

        # Resample to exactly 16 points
        num_drawn_points = len(self.current_drawing_points)
        indices = np.linspace(0, num_drawn_points - 1, 16, dtype=int)
        resampled_points_px = [self.current_drawing_points[i] for i in indices]

        # Normalize pixel coordinates to a 0-100 range and invert Y for typical Cartesian
        normalized_coordinates = []
        max_coord_value = 100.0 # Target range for AI model inputs
        for px, py in resampled_points_px:
            norm_x = (px / self.canvas_width) * max_coord_value
            norm_y = ((self.canvas_height - py) / self.canvas_height) * max_coord_value # Invert Y

            normalized_coordinates.append(norm_x)
            normalized_coordinates.append(norm_y)

        # Populate the output entries
        for i, val in enumerate(normalized_coordinates):
            if i < len(self.output_entries): # Ensure we don't go out of bounds
                self.output_entries[i].delete(0, tk.END)
                self.output_entries[i].insert(0, f"{val:.4f}") # Format to 4 decimal places

        messagebox.showinfo("Shape Used", "Drawn shape converted to 16 coordinates and populated output fields.")
        # Preview the shape from the fields (this will also update orientation)
        self.preview_shape_from_fields()

    def preview_shape_from_fields(self):
        """
        Reads the current output (coordinate) entries, validates them,
        and draws the corresponding shape on the canvas. This is for previewing
        the numerical values currently in the fields.
        """
        self._cancel_interpolation() # Stop any active interpolation
        output_values = self._validate_and_get_values(self.output_entries)
        if output_values is None:
            messagebox.showerror("Input Error", "All output (coordinate) fields must contain valid numerical values for preview.")
            self._update_orientation_label("Invalid Field Values")
            return
        if len(output_values) != 32:
            messagebox.showerror("Validation Error", "Please provide exactly 32 output values (16 pairs) for preview.")
            self._update_orientation_label("Invalid # of Output Values")
            return

        # Reshape flat list of 32 values into 16 (x,y) pairs
        coordinates = []
        for i in range(0, len(output_values), 2):
            coordinates.append((output_values[i], output_values[i+1]))

        # Clear any existing drawn shape and redraw the *numerical* shape
        self.clear_canvas()
        self._draw_shape_on_canvas(coordinates, line_color="red", point_color="green") # Use different colors for clarity



        # Set up interpolation
        start_row_index = self.cycle_index if self.cycle_index != -1 else 0
        end_row_index = (start_row_index + 1) % len(self.data_rows)

        start_values = self.data_rows[start_row_index][8:]
        end_values = self.data_rows[end_row_index][8:]


        # Reshape flat list of 32 values into 16 (x,y) pairs
        start_coords = []
        for i in range(0, len(start_values), 2):
            start_coords.append((start_values[i], start_values[i+1]))

        # Reshape flat list of 32 values into 16 (x,y) pairs
        end_coords = []
        for i in range(0, len(end_values), 2):
            end_coords.append((end_values[i], end_values[i+1]))
        print(start_coords)
        for start,end in zip(start_coords,end_coords):
            print(start,end)
            
            p1_x,p1_y = start
            p2_x,p2_y = end
            self._draw_lines_on_canvas([start,end])
            #self.canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill="blue", width=2)


    def _draw_lines_on_canvas(self, coordinates, line_color="blue", point_color="red"):
        """
        Draws the given list of (x,y) coordinates as a closed shape on the canvas.
        Includes scaling and offset to fit the coordinates within the canvas.
        This method now draws from a *given list of coordinates* (which might be
        from field values or normalized drawn points).

        Args:
            coordinates (list of tuple): A list of (x, y) tuples representing the points.
            line_color (str): Color for the lines.
            point_color (str): Color for the points.
        """
        

        min_model_coord = 0.0
        max_model_coord = 100.0

        padded_canvas_margin = 20 # Padding to keep shapes from touching canvas edge
        padded_canvas_width = self.canvas_width - 2 * padded_canvas_margin
        padded_canvas_height = self.canvas_height - 2 * padded_canvas_margin

        # Calculate scale factor ensuring shape fits within padded canvas
        scale_x_padded = padded_canvas_width / (max_model_coord - min_model_coord)
        scale_y_padded = padded_canvas_height / (max_model_coord - min_model_coord)
        scale = min(scale_x_padded, scale_y_padded)

        # Calculate offsets to center the scaled shape within the padded canvas
        # If one dimension scales down more, the other will have extra space.
        center_offset_x = (padded_canvas_width - (max_model_coord - min_model_coord) * scale) / 2
        center_offset_y = (padded_canvas_height - (max_model_coord - min_model_coord) * scale) / 2


        offset_x = padded_canvas_margin + center_offset_x
        offset_y = self.canvas_height - (padded_canvas_margin + center_offset_y) # Tkinter Y is inverted for drawing


        canvas_points = []
        for x_model, y_model in coordinates:
            canvas_x = (x_model - min_model_coord) * scale + offset_x
            canvas_y = offset_y - (y_model - min_model_coord) * scale # Invert Y for drawing on canvas

            canvas_points.append((canvas_x, canvas_y))

            point_radius = 3

        for i in range(len(canvas_points)):
            p1_x, p1_y = canvas_points[i]
            p2_x, p2_y = canvas_points[(i + 1) % len(canvas_points)] # Connect last point to first
            self.canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill=line_color, width=2)

    def _draw_shape_on_canvas(self, coordinates, line_color="blue", point_color="red"):
        """
        Draws the given list of (x,y) coordinates as a closed shape on the canvas.
        Includes scaling and offset to fit the coordinates within the canvas.
        This method now draws from a *given list of coordinates* (which might be
        from field values or normalized drawn points).

        Args:
            coordinates (list of tuple): A list of (x, y) tuples representing the points.
            line_color (str): Color for the lines.
            point_color (str): Color for the points.
        """
        self.clear_canvas() # Ensure canvas is clear before drawing
        if not coordinates:
            self._update_orientation_label("No points to draw")
            return

        # Calculate orientation using the original (model) coordinates
        orientation_text = self._check_orientation(coordinates)
        self._update_orientation_label(orientation_text)

        min_model_coord = 0.0
        max_model_coord = 100.0

        padded_canvas_margin = 20 # Padding to keep shapes from touching canvas edge
        padded_canvas_width = self.canvas_width - 2 * padded_canvas_margin
        padded_canvas_height = self.canvas_height - 2 * padded_canvas_margin

        # Calculate scale factor ensuring shape fits within padded canvas
        scale_x_padded = padded_canvas_width / (max_model_coord - min_model_coord)
        scale_y_padded = padded_canvas_height / (max_model_coord - min_model_coord)
        scale = min(scale_x_padded, scale_y_padded)

        # Calculate offsets to center the scaled shape within the padded canvas
        # If one dimension scales down more, the other will have extra space.
        center_offset_x = (padded_canvas_width - (max_model_coord - min_model_coord) * scale) / 2
        center_offset_y = (padded_canvas_height - (max_model_coord - min_model_coord) * scale) / 2


        offset_x = padded_canvas_margin + center_offset_x
        offset_y = self.canvas_height - (padded_canvas_margin + center_offset_y) # Tkinter Y is inverted for drawing


        canvas_points = []
        for x_model, y_model in coordinates:
            canvas_x = (x_model - min_model_coord) * scale + offset_x
            canvas_y = offset_y - (y_model - min_model_coord) * scale # Invert Y for drawing on canvas

            canvas_points.append((canvas_x, canvas_y))

            point_radius = 3
            self.canvas.create_oval(canvas_x - point_radius, canvas_y - point_radius,
                                    canvas_x + point_radius, canvas_y + point_radius,
                                    fill=point_color, outline=point_color)

        for i in range(len(canvas_points)):
            p1_x, p1_y = canvas_points[i]
            p2_x, p2_y = canvas_points[(i + 1) % len(canvas_points)] # Connect last point to first
            self.canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill=line_color, width=2)

    def _check_orientation(self, coordinates):
        """
        Determines if a polygon's vertices are ordered clockwise or anti-clockwise (counter-clockwise).
        Uses the sum of the signed areas method.
        Assumes coordinates are in a standard Cartesian system (Y increases upwards).

        Args:
            coordinates (list of tuple): A list of (x, y) tuples representing the vertices.

        Returns:
            str: "Clockwise", "Anti-clockwise", "Collinear/Invalid", or "Too few points".
        """
        if len(coordinates) < 3:
            return "Too few points"

        # Calculate the sum of the cross products (signed area)
        # Sum (xi * yi+1 - xi+1 * yi) for i = 0 to n-1
        # where (xn, yn) is (x0, y0)
        signed_area_sum = 0
        for i in range(len(coordinates)):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[(i + 1) % len(coordinates)] # Next point, wraps around
            signed_area_sum += (x1 * y2) - (x2 * y1)

        # For a simple polygon:
        # If sum > 0, it's anti-clockwise (counter-clockwise)
        # If sum < 0, it's clockwise
        # If sum == 0, it's collinear or self-intersecting (for convex, collinear)
        if signed_area_sum > 0:
            return "Anti-clockwise"
        elif signed_area_sum < 0:
            return "Clockwise"
        else:
            return "Collinear/Invalid" # Or specifically "Collinear" for simple cases


    def _update_orientation_label(self, text):
        """Updates the text of the orientation label."""
        self.orientation_label.config(text=f"Orientation: {text}")

    def _load_data_from_csv(self, filename):
        """
        Loads data from the specified CSV file into self.data_rows.
        """
        try:
            with open(filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                loaded_rows = []
                for row in reader:
                    try:
                        # Convert all values in the row to floats
                        float_row = [float(item) for item in row]
                        # Basic validation for row length (8 inputs + 32 outputs)
                        if len(float_row) == 40:
                            loaded_rows.append(float_row)
                        else:
                            print(f"Skipping malformed row with {len(float_row)} columns: {row}")
                    except ValueError as ve:
                        print(f"Skipping row due to non-numeric data: {row} - {ve}")
                self.data_rows.extend(loaded_rows)
            self.update_display()
            if loaded_rows:
                messagebox.showinfo("Data Loaded", f"Successfully loaded {len(loaded_rows)} data rows from '{filename}'.")
            else:
                messagebox.showwarning("Data Loaded", f"'{filename}' found, but no valid data rows were loaded.")
        except FileNotFoundError:
            messagebox.showinfo("Data Load", f"'{filename}' not found. Starting with an empty dataset.")
        except Exception as e:
            messagebox.showerror("Load Error", f"An error occurred while loading data from '{filename}': {e}")


    def update_display(self):
        """
        Updates the scrolled text area with the current accumulated data rows
        and updates the row count label.
        """
        self.data_display.config(state='normal') # Enable editing temporarily
        self.data_display.delete('1.0', tk.END) # Clear existing text
        for i, row in enumerate(self.data_rows):
            inputs = ", ".join([f"{x:.2f}" for x in row[:8]])
            outputs = ", ".join([f"({row[8 + j]:.2f}, {row[8 + j + 1]:.2f})" for j in range(0, 32, 2)])
            # Add a 3-digit row number prefix for easy identification
            self.data_display.insert(tk.END, f"[{i+1:03d}] Inputs [{inputs}] -> Outputs [{outputs}]\n")
        self.data_display.config(state='disabled') # Disable editing again
        self.status_label.config(text=f"Current Data Rows ({len(self.data_rows)}): Click on a row number to select for editing/deletion.")

    def load_selected_row(self):
        """
        Loads the data from the row where the cursor is currently placed in the
        data_display into the input and output entry fields for editing.
        """
        self._cancel_interpolation() # Stop any active interpolation
        try:
            # Get the line number where the cursor is. Tkinter lines are 1-based.
            current_line_index_str = self.data_display.index(tk.INSERT).split('.')[0]
            selected_row_display_num = int(current_line_index_str)
            selected_row_list_index = selected_row_display_num - 1 # Convert to 0-based index

            if 0 <= selected_row_list_index < len(self.data_rows):
                self.editing_row_index = selected_row_list_index
                self.cycle_index = selected_row_list_index # Also set cycle index for continuity
                row_data = self.data_rows[self.editing_row_index]

                self.clear_fields_without_resetting_editing_index() # Clear fields but not editing/cycle index

                self._populate_fields_from_row_data(row_data)

                messagebox.showinfo("Row Loaded", f"Row {selected_row_display_num} loaded for editing. Make changes and click 'Update Loaded Row'.")
                # Preview the loaded shape (this will also update orientation)
                self.preview_shape_from_fields()
            else:
                messagebox.showwarning("Selection Error", "Please place the cursor within a valid data row to load it.")
        except ValueError:
            messagebox.showwarning("Selection Error", "Invalid selection. Please click on a row to load it.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def update_loaded_row(self):
        """
        Updates the row currently loaded for editing with the values from the
        input and output entry fields.
        """
        self._cancel_interpolation() # Stop any active interpolation
        if self.editing_row_index == -1:
            messagebox.showwarning("No Row Loaded", "No row is currently loaded for updating. Please load a row first.")
            return

        input_values = self._validate_and_get_values(self.input_entries)
        if input_values is None:
            messagebox.showerror("Input Error", "All input fields must contain valid numerical values.")
            return

        output_values = self._validate_and_get_values(self.output_entries)
        if output_values is None:
            messagebox.showerror("Input Error", "All output (coordinate) fields must contain valid numerical values.")
            return

        if len(input_values) != 8 or len(output_values) != 32:
            messagebox.showerror("Validation Error", "Inputs must be 8 values and outputs 32 values (16 pairs).")
            return

        new_row = input_values + output_values
        self.data_rows[self.editing_row_index] = new_row
        self.update_display()
        self.clear_fields() # Clear fields after updating
        messagebox.showinfo("Row Updated", f"Row updated successfully!")

    def delete_selected_row(self):
        """
        Deletes the row where the cursor is currently placed in the data_display.
        """
        self._cancel_interpolation() # Stop any active interpolation
        try:
            current_line_index_str = self.data_display.index(tk.INSERT).split('.')[0]
            selected_row_display_num = int(current_line_index_str)
            selected_row_list_index = selected_row_display_num - 1 # Convert to 0-based index

            if 0 <= selected_row_list_index < len(self.data_rows):
                if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete row {selected_row_display_num}?"):
                    del self.data_rows[selected_row_list_index]
                    self.update_display()
                    self.clear_fields() # Clear fields after deletion
                    messagebox.showinfo("Row Deleted", f"Row {selected_row_display_num} deleted successfully.")
            else:
                messagebox.showwarning("Selection Error", "Please place the cursor within a valid data row to delete it.")
        except ValueError:
            messagebox.showwarning("Selection Error", "Invalid selection. Please click on a row to delete it.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def flip_shape_orientation(self):
        """
        Flips the orientation of the shape by reversing the order of the coordinate pairs
        in the output entry fields.
        """
        self._cancel_interpolation() # Stop any active interpolation
        output_values = self._validate_and_get_values(self.output_entries)
        if output_values is None:
            messagebox.showerror("Flip Error", "Output fields must contain valid numerical values to flip orientation.")
            return
        if len(output_values) != 32:
            messagebox.showerror("Flip Error", "Expected 32 output values (16 pairs) to flip orientation.")
            return

        # Convert flattened list to (x,y) tuples
        coordinates = []
        for i in range(0, len(output_values), 2):
            coordinates.append((output_values[i], output_values[i+1]))

        # Reverse the order of the coordinate pairs
        coordinates.reverse()

        # Flatten back to a list of 32 values
        flipped_output_values = []
        for x, y in coordinates:
            flipped_output_values.append(x)
            flipped_output_values.append(y)

        # Populate the output entries with the new, flipped values
        for i, val in enumerate(flipped_output_values):
            self.output_entries[i].delete(0, tk.END)
            self.output_entries[i].insert(0, f"{val:.4f}")
        
        messagebox.showinfo("Orientation Flipped", "Shape orientation has been flipped.")
        self.preview_shape_from_fields() # Redraw with new orientation and update label

    def _cancel_interpolation(self):
        """Cancels any ongoing interpolation animation."""
        if self.interpolation_job:
            self.master.after_cancel(self.interpolation_job)
            self.interpolation_job = None
        self.interpolating = False
        self.start_interpolation_coords_flat = None
        self.end_interpolation_coords_flat = None
        self.current_interpolation_step = 0 # Reset step

    def _start_cycle_interpolation(self):
        """
        Initiates the cycling and interpolation animation between data rows.
        """
        if not self.data_rows:
            messagebox.showwarning("No Data", "No data rows available to cycle through. Please add some data first.")
            return

        # Cancel any ongoing interpolation before starting a new one
        self._cancel_interpolation()
        
        # Determine the start and end rows for interpolation
        if len(self.data_rows) == 1:
            # If only one row, just display it without interpolation
            self.cycle_index = 0
            self.editing_row_index = 0
            self._populate_fields_from_row_data(self.data_rows[0])
            self.preview_shape_from_fields() # This will also update orientation
            messagebox.showinfo("Cycle View", f"Displaying Row 1 of 1.")
            return

        # Set up interpolation
        start_row_index = self.cycle_index if self.cycle_index != -1 else 0
        end_row_index = (start_row_index + 1) % len(self.data_rows)

        self.start_interpolation_coords_flat = self.data_rows[start_row_index][8:]
        self.end_interpolation_coords_flat = self.data_rows[end_row_index][8:]

        self.cycle_index = end_row_index # Update cycle index to the target row for next cycle
        self.editing_row_index = self.cycle_index # Set editing index to the target row

        self.current_interpolation_step = 0
        self.interpolating = True
        
        # Clear fields and canvas before starting interpolation
        self.clear_fields_without_resetting_editing_index() 
        
        messagebox.showinfo("Cycle View", f"Interpolating from Row {start_row_index + 1} to Row {end_row_index + 1}.")
        self._interpolate_frame()

    def _interpolate_frame(self):
        """
        Calculates and draws a single frame of the interpolation animation.
        Schedules the next frame until interpolation is complete.
        """
        if not self.interpolating or self.start_interpolation_coords_flat is None or self.end_interpolation_coords_flat is None:
            return # Ensure interpolation is active and data is set

        if self.current_interpolation_step <= self.interpolation_steps:
            t = self.current_interpolation_step / self.interpolation_steps
            
            # Linear interpolation for each coordinate
            interpolated_coords_flat = [
                s + (e - s) * t
                for s, e in zip(self.start_interpolation_coords_flat, self.end_interpolation_coords_flat)
            ]

            # Convert flattened list to (x,y) tuples for drawing
            interpolated_coords_tuples = []
            for i in range(0, len(interpolated_coords_flat), 2):
                interpolated_coords_tuples.append((interpolated_coords_flat[i], interpolated_coords_flat[i+1]))

            self.clear_canvas()
            # Use distinct colors for interpolation animation to differentiate from static previews
            # The _draw_shape_on_canvas function will call _check_orientation for each frame
            self._draw_shape_on_canvas(interpolated_coords_tuples, line_color="darkgreen", point_color="blue") 

            self.current_interpolation_step += 1
            self.interpolation_job = self.master.after(self.interpolation_delay_ms, self._interpolate_frame)
        else:
            # Interpolation finished
            self._cancel_interpolation() # Reset flags and job
            # Ensure the final target shape is fully drawn and its numbers are in fields
            self._populate_fields_from_row_data(self.data_rows[self.cycle_index])
            self.preview_shape_from_fields() # Redraw final shape clearly with its dedicated preview colors


    def save_to_csv(self):
        """
        Saves the accumulated data_rows to a CSV file named 'training_data.csv'.
        """
        self._cancel_interpolation() # Stop any active interpolation
        if not self.data_rows:
            messagebox.showwarning("No Data", "No data rows to save. Please add some rows first.")
            return

        filename = 'training_data.csv'
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.data_rows)
            messagebox.showinfo("Save Success", f"Data successfully saved to '{filename}'")
        except IOError as e:
            messagebox.showerror("File Error", f"Could not save file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingDataGeneratorUI(root)
    root.mainloop()
