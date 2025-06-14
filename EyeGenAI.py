import numpy as np

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
        # First moment (mean of gradients)
        m_weights = np.zeros_like(self.weights)
        m_biases = np.zeros_like(self.biases)
        # Second moment (uncentered variance of gradients)
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

            # Update biased first moment estimates (m_t)
            m_weights = beta1 * m_weights + (1 - beta1) * grad_weights
            m_biases = beta1 * m_biases + (1 - beta1) * grad_biases

            # Update biased second moment estimates (v_t)
            v_weights = beta2 * v_weights + (1 - beta2) * (grad_weights ** 2)
            v_biases = beta2 * v_biases + (1 - beta2) * (grad_biases ** 2)

            # Compute bias-corrected first moment estimates (m_hat)
            m_weights_hat = m_weights / (1 - beta1 ** t)
            m_biases_hat = m_biases / (1 - beta1 ** t)

            # Compute bias-corrected second moment estimates (v_hat)
            v_weights_hat = v_weights / (1 - beta2 ** t)
            v_biases_hat = v_biases / (1 - beta2 ** t)

            # Update weights and biases using Adam rule
            self.weights -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
            self.biases -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

            if (epoch + 1) % (epochs // 100 or 1) == 0: # Print loss periodically
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

        print(f"--- Training Finished. Final Loss: {loss:.6f} ---")


# --- Example Usage ---
if __name__ == "__main__":
    # Create an instance of the AI
    coordinate_ai = CoordinateGeneratorAI()
    #import random
    #print(int(np.random.rand() * (2**32 - 1)))
    #np.random.seed(90)
    #1123463763
    #1123463763
    print(coordinate_ai.weights.tolist())
    coordinate_ai.weight = np.array([[-0.28153669515806523, 0.27074768370333757, 0.159261293134219, 0.28856091063353384, 0.27472520803475664, -0.37941748006315296, -0.1459147249430772, 0.29102860551884613, -0.32642170899907036, -0.2803896881787261, 0.303652037175728, -0.19886412429253997, 0.20871564864463543, -0.22305355375431604, -0.23321187951797395, -0.31228552277882543, 0.1682128254939913, -0.15612357223575232, 0.26210626398227777, -0.0687103429728132, -0.025140651008098924, 0.13616759640370346, -0.2271288668895816, -0.18810508060051045, 0.13947903189178523, -0.2247311769709677, -0.3076145529580425, -0.2638158248414441, -0.0436945865189291, 0.36913312875180926, -0.29175638585587094, -0.2760044069120462], [-0.25645357458753537, 0.05980393257279398, 0.015494034024247483, -0.1703232886055218, -0.318413307233093, -0.21372624032657683, 0.0663066306467614, 0.30144100392243367, 0.05636999837335227, -0.1674495953487106, 0.21281134262852375, 0.3054950016997662, -0.01090134760297895, -0.3702189510401145, -0.031180736650795238, -0.36935376192768826, 0.17909984757552155, 0.07316507821705265, -0.2688858251617866, -0.1651921860939983, -0.15204725658732754, -0.044084243671518364, -0.19222911044959728, -0.006167141658518005, 0.14621513591752655, 0.06278334668296437, -0.3049406235633192, -0.10850181612464993, -0.09503290774237272, -0.05171013959258697, -0.22071035984460421, -0.003669751882174943], [-0.2866161709186751, -0.03354654370445692, 0.05452757067293168, 0.1653295946401272, -0.004747759070388047, -0.06465368023949941, -0.16285679097152805, 0.1546328800452903, 0.3627415147961931, 0.3613963009587803, -0.19816435863950999, -0.1861313390103676, -0.24645326412067672, -0.2669415068952537, 0.1744668794277514, -0.2891258354691505, 0.14605383308434572, -0.08034510007951445, 0.35090988455607164, 0.07859312341762026, 0.3267239007093782, -0.06920345225590291, 0.25285844651074196, 0.12610698979123147, 0.17598030541449705, 0.29568289436183925, 0.33060622129158446, -0.21963333393782375, -0.3619383648809395, 0.264064548576587, 0.06303755392935534, 0.2399612311374527], [0.03190541913595629, -0.32670161863551456, 0.31989719007391626, 0.14056028881529437, 0.35052013933841364, -0.13446528962663173, -0.10116843161672906, 0.17609998494796175, -0.1367282560195595, -0.0022520573254684595, -0.04428818348216024, -0.036660271547740875, -0.34405625783354415, 0.021677674347790887, -0.2519928171764385, 0.11761495682718437, -0.22660603481543612, -0.09741055737794391, 0.20741140302843197, -0.018047289845240455, -0.026029105503174887, -0.21168218148425186, -0.08208309690877952, 0.21861697384073808, 0.006852775369290853, 0.28672370393009816, 0.38654262015342555, -0.1959587278465816, 0.06474071016576283, -0.36319838123937076, -0.27489887001798147, 0.35163700341785775], [0.20713258891590158, -0.10916274304076745, 0.3133428194374227, -0.3481987564133794, -0.11084007727323014, -0.06498877778468953, 0.09427659906379754, -0.08540214799231655, 0.06997937256735148, -0.24796732131604754, 0.0519217525738942, -0.2578830860612654, -0.2464594721812113, -0.38714224106772255, 0.07512901846688441, -0.30981060210115147, 0.06826322461397427, -0.11625783581529225, -0.2894610008976575, 0.15835351219999316, -0.008559471525970752, 0.37027701642329935, -0.18993498560534364, -0.08656458349671148, 0.3595294638494514, 0.04140980950211309, -0.07650327792125688, 0.3833638244777583, 0.021381580356891083, 0.37004460819441853, 0.2549436455015801, -0.0008349151036415314], [-0.06968392435815601, -0.12291638934528798, -0.1387436922344346, -0.04195667513863138, -0.04932877559117893, 0.10753873426694172, 0.1303467458927049, -0.16270612002170054, -0.31173651984235357, -0.2983736415417556, 0.3036100414956351, -0.24757937209906583, 0.14229072706948664, -0.29780030116257317, -0.17578634120636175, 0.026614962139225984, 0.23196044229800206, -0.18537942107796798, -0.08729579325110093, -0.36767323258233436, -0.2615867712644543, 0.23397714315830354, -0.3676879860054829, 0.058516224499789216, -0.367435894816508, -0.2807889127978985, -0.02870359309332915, -0.024507763739280763, 0.3862543521939712, -0.34229639366887554, 0.34963286276270134, 0.3758596214220452], [-0.0021381773118236413, 0.34243403501337843, 0.0888324793229589, -0.11221668837287152, 0.0838702552129541, 0.11683010843081498, -0.20776012698531224, -0.25983945374999573, 0.20716564444481522, 0.09357167508147968, 0.007339714567326738, -0.2585277099954467, 0.26714430677971646, -0.22906568915972517, 0.07017097874493383, 0.032459307066547816, -0.026555598011372805, 0.22961934062557698, -0.022821814472763213, 0.14165399287769342, 0.31844565888621856, -0.1733574478229295, -0.09700493315355246, 0.22425339499357588, 0.223000403290594, 0.2563039745757968, 0.3514137024816604, -0.2506123386265551, 0.14663638780990418, 0.013274104030171818, -0.15684322588415606, 0.3284486271953445], [0.05778827722059443, -0.004380980414699254, -0.26484408521258496, -0.2188206074357804, 0.10620461988449276, 0.27411436265465616, 0.08549366770985628, -0.27122600918337036, 0.2658373052940236, -0.15401331323018466, -0.01572749872270246, -0.3587127622396149, 0.1132238475559505, -0.2757829783564573, 0.1750511485198225, -0.345808821252161, -0.14197366157790747, -0.3604885119252528, -0.31917985712469515, 0.08361131459983562, -0.3656270164666979, 0.383888046032707, -0.22051467843712058, -0.048320270939797294, 0.019659368217922468, 0.3423645623463777, -0.32173287133556405, 0.336666562611012, 0.13285299453426302, 0.29896675192031263, 0.3631264821818676, -0.12708350979191507]])


    coordinate_ai.weight = np.array([[4.30942308662026, 0.7330169134526499, 7.199831399175903, 1.0155379691220672, -0.3445386752511379, 3.782160276732871, 21.22827887157904, 3.851956489061954, 23.91386922526851, 2.3481002655906797, 23.7022797707597, 2.260644434398852, 21.243906972429624, 6.9674712553545675, 20.6084396200228, 13.247180534067693, 16.848334542461544, 17.436650326481587, 10.766265721299607, 18.616378153363776, 6.9527176409517075, 17.291719770908426, -0.016805968172373287, 13.753121223792075, 1.5877193369724898, 9.800745815311528, 1.4735442926674118, 5.092274886953538, 2.588680688397993, 0.7805529862310807, 4.589082158676313, 2.173163874386106], [0.3470404142711523, -0.24119375176976748, 0.0038545054809779322, 0.22601890668333913, -0.2149902134578006, -0.15452236078777257, -0.16595353181151423, -0.24062757367727483, 0.38113373461563005, -0.30624054048227783, 0.11810330729275853, -0.05458286237107063, -0.21094535359039354, -0.031496340972994585, 0.08704144770130734, 0.04909909399763901, 0.1583186275976679, -0.26993688431889384, 0.30983761020888234, 0.29761838097201054, 0.09709632492721426, 0.07651888421565273, -0.025594595312335655, -0.1132962070815779, -0.3144602774220114, 0.20141769477329652, 0.24199761112712592, 0.2631318312827342, 0.027049432782579597, 0.17116318718752077, -0.13467108082218554, 0.27760363617902195], [10.92021670882184, 19.755500065719392, 13.178640186416608, 20.647473022068436, -0.28000575711875386, 22.359600684844587, 14.602567107542068, 24.360062057158647, 16.232870862818036, 25.02881113355469, 18.539447246812077, 25.841597772126754, 21.12876244833317, 26.27820621598817, 23.926443326577953, 23.725431225693704, 25.241995619952004, 18.427452169122436, 24.39791026175975, 13.318882613592843, 22.869200505845548, 10.56014191314972, -0.0879412581380789, 9.271609151414147, 16.857066348513488, 10.174400912368556, 14.76518479739123, 12.638975853627034, 12.765037441563964, 15.957026034829022, 10.85049110457053, 20.252006861819524], [-0.190111757369476, -0.3358993098514714, 0.0042576622173561285, -0.33453858717844986, -0.032919945951980834, 0.1399805353319976, -0.34869500882145565, 0.22690661340450424, 0.2502568766829094, -0.09321231516871142, 0.01408955969412451, 0.14013588999415505, -0.17319009200638774, -0.0305553347572608, 0.3022345072976562, -0.06416145683842056, -0.09629451025655722, -0.28688758338646625, -0.03990313602858375, -0.09623650843830583, 0.015266845241849347, 0.19491122244154224, -0.3666163301399595, 0.3851256605823473, 0.04800907135216054, 0.03022536654788266, 0.15264664030452213, 0.254463612597975, 0.2358658117722301, 0.32776511615600923, -0.07976870812578901, 0.016460213844229188], [4.732608865800373, 10.42356053186222, 8.700295778894834, 11.529555712143607, -0.11725452866519676, 13.269188909980453, 21.373306389033967, 12.514230044270642, 24.879029328277543, 9.702497183853305, 26.201703043838062, 7.9297020435230445, 25.888933025953865, 8.944813888087872, 24.91830422855215, 15.790536936198146, 21.05644375744008, 19.879122829181004, 17.331708762040748, 20.69184807739602, 12.991401115901601, 19.88244753984545, -0.06051536411599556, 18.27817783547875, 8.138158940768193, 15.662306202660327, 5.26061899646315, 13.16646349202357, 4.094482304156284, 11.058094434339036, 4.459328214463993, 11.069485066408754], [3.964260834371235, -0.8125901650536256, 5.975402314291816, -0.9737400794648184, 0.26344035472019056, 1.755144105917954, 20.392632848330564, 1.915392814598578, 23.090875165255675, 1.3795014081887147, 22.62840971004675, 3.166202889394392, 20.416218067658, 10.839299587745852, 19.858789181368433, 12.32363705185058, 16.088107188642898, 16.85209411505544, 9.672286114593343, 17.717453040662456, 5.80886456094613, 16.542498553197053, 0.21499572053397098, 12.622814359051635, -0.13435616750989582, 8.39952090957845, 0.8357542435665514, 3.1743677330506173, 1.9326536270989472, -1.2459796594748949, 5.404833408604671, -0.037963080599529145], [10.717194675814424, 19.851061720770577, 13.289680673057422, 19.94335511539533, -0.23477033127504351, 22.13423986308943, 15.136201600973337, 24.448171552780973, 16.433095172488006, 24.69236965429137, 18.87035963535802, 25.79429610223765, 21.179127516251988, 26.7904571160592, 23.52518597995102, 23.86542562517568, 25.603046473256335, 18.47705832142491, 24.5094974581314, 13.72719586488335, 23.075244839219835, 10.41785744963317, 0.2867052761636909, 8.663357783052597, 17.277445240507127, 9.961419455443021, 15.00430116803361, 13.039066002507118, 12.924944528737626, 15.454045447183505, 10.965369683755526, 20.22956710288197], [9.33773299234397, 21.874948504234816, 13.167595535915087, 23.63729916512447, -0.3321915301559035, 24.367889903203583, 22.606028741578083, 23.91826063610311, 27.246890501924188, 22.292988361216025, 29.936084924554223, 20.151992601772914, 31.533391200120025, 18.11602346443674, 29.72628753925161, 21.958202041184688, 27.261822437287535, 23.29358068570741, 24.789062732447434, 23.510779540596076, 21.932313114082213, 23.634510927036533, -0.307753843969994, 24.184815023578707, 16.362223044825694, 23.874237657505432, 13.342198585281531, 23.867008615979987, 10.389215595181662, 23.318565835956825, 8.542187035393406, 22.200697630341857]])
    coordinate_ai.biases = np.array([11.181264962847557, 21.710654112172126, 13.502391095737051, 23.401119990602332, 17.692114626932575, 0.0, 0.0, 24.024883337317714, 27.039969983958894, 24.836267349407226, 0.0, 0.0, 31.396113471971866, 0.0, 29.55696044979038, 23.577494380670558, 27.212291458717317, 0.0, 25.00569846985304, 0.0, 0.0, 23.99811443361853, 20.01883785018025, 24.37500607296006, 0.0, 24.012825392055444, 14.783688797341956, 23.723695685015365, 0.0, 23.14482875708152, 11.077262565242167, 22.073743851659284])


    coordinate_ai.biases = np.array([16.89897364525974, 33.39987157423893, 20.405892157536265, 36.11636778547865, 27.453360893491183, 24.05488412868125, 22.438669248583736, 36.86303921013795, 42.93692286094244, 38.03263709230563, 30.175951472031475, 25.569132925024537, 47.93751167159797, 26.33498277963107, 45.09236359695944, 36.0226689702753, 41.27773308892646, 23.093497592485882, 38.26309878305495, 23.564871284109948, 22.796245498415225, 36.35962927299558, 30.637034476140162, 36.82359277987591, 16.872505541191916, 36.526957842057485, 22.483447812677113, 36.209360993008694, 12.606088816093473, 35.5844257076049, 16.81905598100475, 34.00594230992297])
    # --- Load Training Data from File ---
    # Please ensure you have a CSV file named 'training_data.csv'
    # in the same directory as this script.
    # The file should contain 40 columns:
    #   - First 8 columns for inputs (X)
    #   - Next 32 columns for outputs (y_flat)
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
        # Use a slightly smaller learning rate with Adam, as it's often more stable
        #coordinate_ai.train(X_train, y_train, epochs=100000, learning_rate=0.0001)
        #coordinate_ai.train(X_train, y_train, epochs=300000, learning_rate=0.0001)
        
        #coordinate_ai.train(X_train, y_train, epochs=400000, learning_rate=0.0001)
        coordinate_ai.train(X_train, y_train, epochs=200000, learning_rate=0.0001)
        
        #coordinate_ai.train(X_train, y_train, epochs=400000, learning_rate=0.000005)

        # --- Test with new, unseen data after training ---
        print("\n--- Testing Trained AI with New Inputs ---")
        new_inputs = np.array([
            [1.5, 2.3, 0.8, 4.1, 5.0, 6.7, 7.2, 8.9],
            [9.1, 1.2, 3.4, 5.6, 7.8, 0.9, 2.1, 4.3]
        ])

        new_inputs = np.array([1.0,0.0,0.0,0.0,0.8,0.1,0.0,0.0])

        predicted_coordinates = coordinate_ai.predict(new_inputs)
        print(coordinate_ai.biases.tolist())
        import turtle

        import math
        z = 0
        while True:
            
            turtle.tracer(1)
            z += math.radians(1)
            new_inputs = np.array([abs(math.sin(z)),0.0,0.0,0.0,0.8,0.1,0.0,0.0])
            new_inputs = np.array([1.0,0.0,0.0,0.0,abs(math.sin(z))*0.8,0.1,0.0,0.0])
            predicted_coordinates = coordinate_ai.predict(new_inputs)
            turtle.tracer(0)
            turtle.clear()
            turtle.penup()

                
            
            if predicted_coordinates is not None:
                #print("\nPredicted 16 (x, y) Coordinate Pairs for New Inputs:")
                for sample_idx, sample_coords in enumerate(predicted_coordinates):
                    #print(f"\nSample {sample_idx + 1}:")
                    for i, coord in enumerate(sample_coords):
                        x,y = int(coord[0]),int(coord[1])
                        turtle.goto(x,y)
                        turtle.pendown()
                        turtle.dot()
                        #print(f"  Pair {i+1}: ({coord[0]:.4f}, {coord[1]:.4f})")
            turtle.update()
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        print("Please create a CSV file with 8 input columns and 32 output columns for training.")
        print("For example, create a file named 'training_data.csv' with comma-separated values like:")
        print("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 1.0,1.1,1.2,1.3, ..., 1.32")
        print("0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 2.0,2.1,2.2,2.3, ..., 2.32")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")


    # Example with incorrect input size during prediction
    print("\n--- Testing prediction with incorrect input size ---")
    incorrect_inputs_pred = [1, 2, 3, 4, 5] # Only 5 inputs
    coordinate_ai.predict(incorrect_inputs_pred)
