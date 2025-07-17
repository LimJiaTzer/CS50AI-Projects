import tensorflow as tf
import numpy as np
import cv2
import os
import sys # To print progress

# --- Configuration ---
# Path to your saved traffic sign model
model_path = r"C:\Users\limji\Documents\Comp sci\Python\CS50_AI_course\traffic\model.keras"

# --- !!! IMPORTANT: SET THESE VARIABLES !!! ---
# Path to the FOLDER containing the images you want to test
# Example: If you have test images for category 5 in a folder named '5'
test_folder_path = r"C:\Users\limji\Documents\Comp sci\Python\CS50_AI_course\traffic\gtsrb\1" # <--- CHANGE THIS

# The TRUE category index for ALL images in the test_folder_path
# This should match the folder name (e.g., 5 for a folder named '5')
true_category_index = 1 # <--- CHANGE THIS (must be an integer)
# --- End of variables to set ---

# Image dimensions expected by the traffic model (must match training)
IMG_HEIGHT = 30
IMG_WIDTH = 30
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_CATEGORIES = 43 # Keep for context if needed

# Optional: Define labels for better output (order must match training folders 0-42)
SIGN_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

# --- Check if paths exist ---
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()
if not os.path.isdir(test_folder_path): # Check if it's a directory
    print(f"Error: Test folder not found or is not a directory: {test_folder_path}")
    exit()
if not isinstance(true_category_index, int) or not (0 <= true_category_index < NUM_CATEGORIES):
     print(f"Error: true_category_index ({true_category_index}) must be an integer between 0 and {NUM_CATEGORIES-1}")
     exit()

# --- Load Model ---
print(f"Loading model from {model_path}...")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    # model.summary() # Optional
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Process Images in Folder ---
print(f"\nProcessing images in folder: {test_folder_path}")
print(f"Expecting all images to be category: {true_category_index} ({SIGN_NAMES[true_category_index] if true_category_index < len(SIGN_NAMES) else 'Unknown Name'})")

correct_predictions = 0
total_images = 0
processed_count = 0
incorrect_predictions = []
image_files = [f for f in os.listdir(test_folder_path) if os.path.isfile(os.path.join(test_folder_path, f))]
num_files_to_process = len(image_files)
print(f"Found {num_files_to_process} files to process.")

for filename in image_files:
    image_path = os.path.join(test_folder_path, filename)
    processed_count += 1
    # Update progress on the same line
    progress_percent = (processed_count / num_files_to_process) * 100
    sys.stdout.write(f"\rProcessing file {processed_count}/{num_files_to_process} ({progress_percent:.1f}%)... ")
    sys.stdout.flush()

    try:
        # --- Image Processing using cv2 (Matches traffic.py) ---
        # 1. Load image using cv2
        image = cv2.imread(image_path)
        if image is None:
            # Handle case where image couldn't be read
            print(f"\nWarning: Could not read image file {filename}. Skipping.")
            continue # Skip to the next file

        # 2. Resize image using cv2 (expects width, height)
        resized_image = cv2.resize(image, IMG_SIZE)

        # 3. Convert to float32 and normalize (matching traffic.py's load_data)
        img_array = resized_image.astype(np.float32) / 255.0

        # 4. Add batch dimension (model expects batch_size, height, width, channels)
        img_batch = np.expand_dims(img_array, axis=0)
        # --- End of cv2 processing ---

        # 5. Make Prediction
        predictions = model.predict(img_batch, verbose=0) # verbose=0 prevents predict() logs for each image
        predicted_class_index = np.argmax(predictions[0])

        # 6. Compare with true label
        if predicted_class_index == true_category_index:
            correct_predictions += 1
        else:
            incorrect_predictions.append(filename)

        total_images += 1 # Increment count of successfully processed images

    except Exception as e:
        # Print error for specific file but continue with others
        sys.stdout.write("\n") # Move to next line after progress indicator
        print(f"Warning: Could not process file {filename}. Error: {e}")
        # Ensure progress indicator continues correctly on next iteration
        if num_files_to_process > 0:
                sys.stdout.write(f"\rProcessing file {processed_count}/{num_files_to_process} ({progress_percent:.1f}%)... ")
                sys.stdout.flush()

# --- Calculate and Print Accuracy ---
sys.stdout.write("\n") # Ensure final output starts on a new line
print("\n--- Results ---")
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print(f"Processed: {total_images} images from folder '{os.path.basename(test_folder_path)}'")
    print(f"Correctly Predicted (as Category {true_category_index}): {correct_predictions}")
    print(f"Accuracy for this folder: {accuracy:.2f}%")
    print(incorrect_predictions)
else:
    print(f"No images were successfully processed in the folder: {test_folder_path}")

