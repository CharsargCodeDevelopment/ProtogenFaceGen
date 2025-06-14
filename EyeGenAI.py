import numpy as np
import tkinter as tk
import math
from tkinter import Scale, Frame, Label

class CoordinateGeneratorAI:
    """
    A simple AI model that generates 16 (x, y) coordinate pairs from 8 input values.
    This updated version includes capabilities for learning from data using the Adam optimizer.

    The model uses a basic feedforward neural network structure with a single
    output layer, an activation function, a loss function, and an optimization
    method for training.
    """

    def __init__(self, input_size=8, output_size=32):
        """
        Initializes the AI model with random weights and biases.

        Args:
            input_size (int): The number of input features (default is 8).
            output_size (int): The total number of output values (default is 32,
                               which translates to 16 (x,y) pairs).
        """
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights randomly using Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (self.input_size, self.output_size))

        # Initialize biases with zeros
        self.biases = np.zeros(self.output_size)

        print(f"AI initialized with input size: {self.input_size}, output size: {self.output_size}")
        print(f"Weights shape: {self.weights.shape}")
        print(f"Biases shape: {self.biases.shape}")

    def _relu(self, x):
        """
        Rectified Linear Unit (ReLU) activation function.
        Returns x if x > 0, otherwise returns 0.
        Introduces non-linearity to the model.
        """
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """
        Derivative of the ReLU activation function.
        Used during backpropagation to calculate gradients.
        Returns 1 if x > 0, otherwise returns 0.
        """
        return (x > 0).astype(float)

    def predict(self, inputs):
        """
        Generates 16 (x, y) coordinate pairs based on the given 8 inputs.

        The process involves:
        1. Performing a dot product of the inputs with the weights.
        2. Adding the biases to the result.
        3. Applying the ReLU activation function.
        4. Reshaping the 32 output values into 16 (x, y) pairs.

        Args:
            inputs (list or numpy.ndarray): A list or numpy array containing 8 numerical inputs.
                                          Can be a single input (1D array) or multiple inputs (2D array).

        Returns:
            numpy.ndarray: A 2D numpy array of shape (N, 16, 2) where N is the
                           number of input samples, and each (16, 2) array
                           represents 16 (x, y) coordinate pairs for a sample.
                           Returns None if the input size is incorrect.
        """
        inputs = np.array(inputs)

        # Handle single input vs. multiple inputs
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1) # Reshape to (1, input_size) for consistent matrix multiplication

        # Validate input size
        if inputs.shape[1] != self.input_size:
            print(f"Error: Expected {self.input_size} inputs per sample, but got {inputs.shape[1]}.")
            return None

        # Step 1: Linear transformation (inputs * weights)
        self.z = np.dot(inputs, self.weights) # Store for backpropagation

        # Step 2: Add biases
        self.activated_outputs = self.z + self.biases

        # Step 3: Apply ReLU activation function
        output_values = self._relu(self.activated_outputs) # Store activated output for backprop

        # Step 4: Reshape the 32 output values into 16 (x, y) coordinate pairs
        coordinates = output_values.reshape(inputs.shape[0], 16, 2)

        return coordinates

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Trains the AI model using provided input (X_train) and target (y_train) data
        using the Adam optimizer.

        Args:
            X_train (numpy.ndarray): Training input data, shape (num_samples, 8).
            y_train (numpy.ndarray): Target output coordinates, shape (num_samples, 16, 2).
            epochs (int): The number of times to iterate over the entire training dataset.
            learning_rate (float): The step size for the Adam optimizer.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): A small constant for numerical stability.
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Reshape target coordinates from (num_samples, 16, 2) to (num_samples, 32)
        y_train_flat = y_train.reshape(y_train.shape[0], self.output_size)

        num_samples = X_train.shape[0]

        # Initialize Adam optimizer's moment estimates
        m_weights = np.zeros_like(self.weights)
        m_biases = np.zeros_like(self.biases)
        v_weights = np.zeros_like(self.weights)
        v_biases = np.zeros_like(self.biases)
        
        # Time step counter for bias correction
        t = 0

        print(f"\n--- Starting Training (Adam Optimizer) ---")
        print(f"Training for {epochs} epochs with learning rate {learning_rate}")

        for epoch in range(epochs):
            t += 1 # Increment time step

            # Forward pass: Calculate predictions
            predictions_raw_output = self._relu(np.dot(X_train, self.weights) + self.biases)

            # Calculate loss (Mean Squared Error)
            loss = np.mean(np.square(predictions_raw_output - y_train_flat))

            # Backward pass: Calculate gradients
            d_output = (predictions_raw_output - y_train_flat) / num_samples
            d_z = d_output * self._relu_derivative(np.dot(X_train, self.weights) + self.biases)

            # Gradients for weights and biases
            grad_weights = np.dot(X_train.T, d_z)
            grad_biases = np.sum(d_z, axis=0)

            # Update Adam optimizer's moment estimates
            m_weights = beta1 * m_weights + (1 - beta1) * grad_weights
            m_biases = beta1 * m_biases + (1 - beta1) * grad_biases
            v_weights = beta2 * v_weights + (1 - beta2) * (grad_weights ** 2)
            v_biases = beta2 * v_biases + (1 - beta2) * (grad_biases ** 2)

            # Compute bias-corrected moment estimates
            m_weights_hat = m_weights / (1 - beta1 ** t)
            m_biases_hat = m_biases / (1 - beta1 ** t)
            v_weights_hat = v_weights / (1 - beta2 ** t)
            v_biases_hat = v_biases / (1 - beta2 ** t)

            # Update weights and biases
            self.weights -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
            self.biases -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

            if (epoch + 1) % (epochs // 100 or 1) == 0: # Print loss periodically
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

        print(f"--- Training Finished. Final Loss: {loss:.6f} ---")


# --- Example Usage ---
if __name__ == "__main__":
    # Create an instance of the AI
    coordinate_ai = CoordinateGeneratorAI()
    
    # --- Pre-trained weights and biases can be loaded here for consistent results ---
    # (The original weights and biases from the script are kept for reproducibility)
    coordinate_ai.weights = np.array([[4.30942308662026, 0.7330169134526499, 7.199831399175903, 1.0155379691220672, -0.3445386752511379, 3.782160276732871, 21.22827887157904, 3.851956489061954, 23.91386922526851, 2.3481002655906797, 23.7022797707597, 2.260644434398852, 21.243906972429624, 6.9674712553545675, 20.6084396200228, 13.247180534067693, 16.848334542461544, 17.436650326481587, 10.766265721299607, 18.616378153363776, 6.9527176409517075, 17.291719770908426, -0.016805968172373287, 13.753121223792075, 1.5877193369724898, 9.800745815311528, 1.4735442926674118, 5.092274886953538, 2.588680688397993, 0.7805529862310807, 4.589082158676313, 2.173163874386106], [0.3470404142711523, -0.24119375176976748, 0.0038545054809779322, 0.22601890668333913, -0.2149902134578006, -0.15452236078777257, -0.16595353181151423, -0.24062757367727483, 0.38113373461563005, -0.30624054048227783, 0.11810330729275853, -0.05458286237107063, -0.21094535359039354, -0.031496340972994585, 0.08704144770130734, 0.04909909399763901, 0.1583186275976679, -0.26993688431889384, 0.30983761020888234, 0.29761838097201054, 0.09709632492721426, 0.07651888421565273, -0.025594595312335655, -0.1132962070815779, -0.3144602774220114, 0.20141769477329652, 0.24199761112712592, 0.2631318312827342, 0.027049432782579597, 0.17116318718752077, -0.13467108082218554, 0.27760363617902195], [10.92021670882184, 19.755500065719392, 13.178640186416608, 20.647473022068436, -0.28000575711875386, 22.359600684844587, 14.602567107542068, 24.360062057158647, 16.232870862818036, 25.02881113355469, 18.539447246812077, 25.841597772126754, 21.12876244833317, 26.27820621598817, 23.926443326577953, 23.725431225693704, 25.241995619952004, 18.427452169122436, 24.39791026175975, 13.318882613592843, 22.869200505845548, 10.56014191314972, -0.0879412581380789, 9.271609151414147, 16.857066348513488, 10.174400912368556, 14.76518479739123, 12.638975853627034, 12.765037441563964, 15.957026034829022, 10.85049110457053, 20.252006861819524], [-0.190111757369476, -0.3358993098514714, 0.0042576622173561285, -0.33453858717844986, -0.032919945951980834, 0.1399805353319976, -0.34869500882145565, 0.22690661340450424, 0.2502568766829094, -0.09321231516871142, 0.01408955969412451, 0.14013588999415505, -0.17319009200638774, -0.0305553347572608, 0.3022345072976562, -0.06416145683842056, -0.09629451025655722, -0.28688758338646625, -0.03990313602858375, -0.09623650843830583, 0.015266845241849347, 0.19491122244154224, -0.3666163301399595, 0.3851256605823473, 0.04800907135216054, 0.03022536654788266, 0.15264664030452213, 0.254463612597975, 0.2358658117722301, 0.32776511615600923, -0.07976870812578901, 0.016460213844229188], [4.732608865800373, 10.42356053186222, 8.700295778894834, 11.529555712143607, -0.11725452866519676, 13.269188909980453, 21.373306389033967, 12.514230044270642, 24.879029328277543, 9.702497183853305, 26.201703043838062, 7.9297020435230445, 25.888933025953865, 8.944813888087872, 24.91830422855215, 15.790536936198146, 21.05644375744008, 19.879122829181004, 17.331708762040748, 20.69184807739602, 12.991401115901601, 19.88244753984545, -0.06051536411599556, 18.27817783547875, 8.138158940768193, 15.662306202660327, 5.26061899646315, 13.16646349202357, 4.094482304156284, 11.058094434339036, 4.459328214463993, 11.069485066408754], [3.964260834371235, -0.8125901650536256, 5.975402314291816, -0.9737400794648184, 0.26344035472019056, 1.755144105917954, 20.392632848330564, 1.915392814598578, 23.090875165255675, 1.3795014081887147, 22.62840971004675, 3.166202889394392, 20.416218067658, 10.839299587745852, 19.858789181368433, 12.32363705185058, 16.088107188642898, 16.85209411505544, 9.672286114593343, 17.717453040662456, 5.80886456094613, 16.542498553197053, 0.21499572053397098, 12.622814359051635, -0.13435616750989582, 8.39952090957845, 0.8357542435665514, 3.1743677330506173, 1.9326536270989472, -1.2459796594748949, 5.404833408604671, -0.037963080599529145], [10.717194675814424, 19.851061720770577, 13.289680673057422, 19.94335511539533, -0.23477033127504351, 22.13423986308943, 15.136201600973337, 24.448171552780973, 16.433095172488006, 24.69236965429137, 18.87035963535802, 25.79429610223765, 21.179127516251988, 26.7904571160592, 23.52518597995102, 23.86542562517568, 25.603046473256335, 18.47705832142491, 24.5094974581314, 13.72719586488335, 23.075244839219835, 10.41785744963317, 0.2867052761636909, 8.663357783052597, 17.277445240507127, 9.961419455443021, 15.00430116803361, 13.039066002507118, 12.924944528737626, 15.454045447183505, 10.965369683755526, 20.22956710288197], [9.33773299234397, 21.874948504234816, 13.167595535915087, 23.63729916512447, -0.3321915301559035, 24.367889903203583, 22.606028741578083, 23.91826063610311, 27.246890501924188, 22.292988361216025, 29.936084924554223, 20.151992601772914, 31.533391200120025, 18.11602346443674, 29.72628753925161, 21.958202041184688, 27.261822437287535, 23.29358068570741, 24.789062732447434, 23.510779540596076, 21.932313114082213, 23.634510927036533, -0.307753843969994, 24.184815023578707, 16.362223044825694, 23.874237657505432, 13.342198585281531, 23.867008615979987, 10.389215595181662, 23.318565835956825, 8.542187035393406, 22.200697630341857]])
    coordinate_ai.biases = np.array([16.89897364525974, 33.39987157423893, 20.405892157536265, 36.11636778547865, 27.453360893491183, 24.05488412868125, 22.438669248583736, 36.86303921013795, 42.93692286094244, 38.03263709230563, 30.175951472031475, 25.569132925024537, 47.93751167159797, 26.33498277963107, 45.09236359695944, 36.0226689702753, 41.27773308892646, 23.093497592485882, 38.26309878305495, 23.564871284109948, 22.796245498415225, 36.35962927299558, 30.637034476140162, 36.82359277987591, 16.872505541191916, 36.526957842057485, 22.483447812677113, 36.209360993008694, 12.606088816093473, 35.5844257076049, 16.81905598100475, 34.00594230992297])

    data_file = 'training_data.csv'
    try:
        data = np.genfromtxt(data_file, delimiter=',')
        X_train = data[:, :8]
        y_train_flat = data[:, 8:]
        num_training_samples = X_train.shape[0]
        y_train = y_train_flat.reshape(num_training_samples, 16, 2)

        print(f"\n--- Loaded {num_training_samples} training samples from {data_file} ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print("\n--- Training AI with Loaded Data ---")
        coordinate_ai.train(X_train, y_train, epochs=200000, learning_rate=0.0001)
        
        # --- TKINTER VISUALIZATION SETUP ---
        print("\n--- Starting Tkinter Visualization ---")
        
        # Tkinter window setup
        CANVAS_WIDTH = 500
        CANVAS_HEIGHT = 500
        DOT_RADIUS = 3
        
        window = tk.Tk()
        window.title("AI Coordinate Generator")

        # Create a main frame to hold the canvas and sliders
        main_frame = Frame(window)
        main_frame.pack(fill="both", expand=True)

        # Create the canvas
        canvas = tk.Canvas(main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
        canvas.pack(side="left", fill="both", expand=True)

        # Create a frame for the sliders
        slider_frame = Frame(main_frame)
        slider_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Create a list to hold the slider widgets
        sliders = []
        for i in range(coordinate_ai.input_size):
            label = Label(slider_frame, text=f"Input {i+1}")
            label.pack()
            # The command is a function that will be called every time the slider moves.
            # The `lambda val: update_frame()` ensures update_frame is called without arguments.
            slider = Scale(slider_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", command=lambda val: update_frame())
            slider.pack(pady=2)
            sliders.append(slider)

        # Set initial values for the sliders for a default view
        sliders[0].set(1.0)
        sliders[4].set(0.8)
        sliders[5].set(0.1)

        def update_frame():
            """
            This function is called whenever a slider is moved.
            It reads the current values from all sliders, gets a new prediction,
            and redraws the canvas.
            """
            # Get the current value from each slider
            new_inputs = np.array([s.get() for s in sliders])
            
            # Get new coordinates from the AI
            predicted_coordinates = coordinate_ai.predict(new_inputs)
            
            # Clear the canvas for the new frame
            canvas.delete("all")

            if predicted_coordinates is not None:
                # We only expect one sample, so we take the first one
                sample_coords = predicted_coordinates[0]
                
                # Flatten the list of coordinates for create_line
                flat_coords = []
                for coord in sample_coords:
                    # Center the drawing by adding half canvas dimensions
                    x = -int(coord[0]) + CANVAS_WIDTH // 2
                    y = -int(coord[1]) + CANVAS_HEIGHT // 2
                    flat_coords.extend([x, y])
                    
                    # Draw a dot at each coordinate
                    canvas.create_oval(
                        x - DOT_RADIUS, y - DOT_RADIUS,
                        x + DOT_RADIUS, y + DOT_RADIUS,
                        fill="blue", outline=""
                    )
                # To connect the dots in order, uncomment the following line
                # canvas.create_line(flat_coords, fill="lightblue")

        # Call update_frame once to draw the initial state from the sliders' default values
        update_frame()
        # Start the Tkinter event loop
        window.mainloop()

    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        print("Please create a CSV file with 8 input columns and 32 output columns for training.")
    except Exception as e:
        print(f"An error occurred: {e}")
