import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
from keras.models import load_model

def create_fading_effect(effects_list, landmarks, start_alpha=1.0, end_alpha=0.0, duration=30):
    if len(landmarks) > 5:
        # Add a new thumbs up effect with its start position at landmark 4
        effects_list.append({
            "position": landmarks[4],
            "alpha": start_alpha,
            "end_alpha": end_alpha,
            "duration": duration,
            "current_frame": 0
        })

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.
    
    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - (y))
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - (x))

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                 alpha_inv * img[y1:y2, x1:x2, c])
        
def load_effect_images():

    # Load your images with effects once, and reuse them to avoid loading them on each frame.
    effects_images = {
        "thumbs up": {
            "image": cv2.imread('thumbs_up.png', cv2.IMREAD_UNCHANGED),
            "alpha": None
        },
        "thumbs down": {
            "image": cv2.imread('thumbs_down.png', cv2.IMREAD_UNCHANGED),
            "alpha": None
        }
    }
    for effect in effects_images.values():
        # Check if the image has an alpha channel
        if effect["image"].shape[2] == 4:
            # If it does, extract it
            effect["alpha"] = effect["image"][..., 3] / 255.0
            effect["image"] = effect["image"][..., :3]
        else:
            # If it doesn't, create a new alpha channel filled with 255 (fully opaque)
            effect["alpha"] = np.ones(effect["image"].shape[:2], dtype=np.float32)
    
    return effects_images

#Function to overlay video frames on top of images
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.
    
    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - (y))
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - (x))

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                 alpha_inv * img[y1:y2, x1:x2, c])



def apply_effect(frame, landmarks, effect_name, effects_images):
    if effect_name not in effects_images:
        return  # Invalid effect name

    # Calculate the height of the thumb in pixels
    thumb_height = int(np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[1])))

    # Resize the effect image to match the height of the thumb
    effect_image = cv2.resize(effects_images[effect_name]["image"], (thumb_height, thumb_height), interpolation = cv2.INTER_AREA)
    effect_alpha = cv2.resize(effects_images[effect_name]["alpha"], (thumb_height, thumb_height), interpolation = cv2.INTER_AREA)

    # Calculate current alpha based on the frame number
    alpha_scale = np.clip(effect_data["current_frame"] / effect_data["duration"], 0, 1)
    current_alpha = (1 - alpha_scale) * effect_data["alpha"] + alpha_scale * effect_data["end_alpha"]
    alpha_mask = effect_alpha * current_alpha

    # Add an offset to the position
    if effect_name == "thumbs up":
        offset = [180, -50]  # Adjust these values as needed
    elif effect_name == "thumbs down":
        offset = [180, -200]
    position_with_offset = [effect_data["position"][0] + offset[0], effect_data["position"][1] + offset[1]]

    overlay_image_alpha(frame, effect_image, position_with_offset, alpha_mask)
    
    # Increment the current frame counter
    effect_data["current_frame"] += 1


def create_bacckground_effect(frame,effect_name):
    global firework_counter
    global balloon_counter
    if effect_name == "fireworks":
        #iterate through the list of fireworks images in the fireworks_frame folder and overlay them on the frame
        overlay_image=cv2.imread("fireworks_frames/frame"+str(firework_counter)+".png",-1)
        overlay_image=cv2.resize(overlay_image,(frame.shape[1],frame.shape[0]))
        #get the alpha channel of the overlay image
        alpha_mask=overlay_image[:,:,3]/255.0
        firework = overlay_image_alpha(frame, overlay_image, (0,0), alpha_mask)
        return firework
        
    if effect_name == "balloons":
        #iterate through the list of balloons images in the balloons_frame folder and overlay them on the frame
        overlay_image=cv2.imread("balloons_frames/frame"+str(balloon_counter)+".png",-1)
        overlay_image=cv2.resize(overlay_image,(frame.shape[1],frame.shape[0]))
        #get the alpha channel of the overlay image
        alpha_mask=overlay_image[:,:,3]/255.0
        balloon = overlay_image_alpha(frame, overlay_image, (0,0), alpha_mask)
        return balloon


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

effects_images = load_effect_images()

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

thumbs_up_effects = []
thumbs_down_effects = []

firework_counter = 0
balloon_counter = 0

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    
    x , y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = [[int(lm.x * x), int(lm.y * y)] for lm in handslms.landmark]


        # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Predict gesture in Hand Gesture Recognition project
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        className = classNames[classID]
        
        # If the gesture is recognized as 'thumbs up', overlay the effect
        if className == 'thumbs up':
            create_fading_effect(thumbs_up_effects, landmarks, start_alpha=1.0, end_alpha=0.0, duration=30)
            for effect_data in thumbs_up_effects[:]:
                apply_effect(frame, landmarks, "thumbs up", effects_images)
                if effect_data["current_frame"] > effect_data["duration"]:
                    thumbs_up_effects.remove(effect_data)

        if className == 'thumbs down':
            create_fading_effect(thumbs_down_effects, landmarks, start_alpha=1.0, end_alpha=0.0, duration=30)
            for effect_data in thumbs_down_effects[:]:
                apply_effect(frame, landmarks, "thumbs down", effects_images)
                if effect_data["current_frame"] > effect_data["duration"]:
                    thumbs_down_effects.remove(effect_data)

        if className == 'rock':
            create_bacckground_effect(frame,"fireworks")
            firework_counter += 2
            if firework_counter > 105:
                firework_counter = 0

        if className != 'rock':
            firework_counter = 0
            
        if className == 'peace':
            create_bacckground_effect(frame,"balloons")
            balloon_counter += 2
            if balloon_counter > 105:
                balloon_counter = 0
            

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2, cv2.LINE_AA)
    # Show the final output
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
            break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()


