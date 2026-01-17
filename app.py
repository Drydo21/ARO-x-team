import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
import math

# ------------------ UI ------------------
st.title("Facial Proportion Analyzer")
st.caption("Based on mathematical facial proportions (Golden Ratio)")

# ------------------ CONSTANT ------------------
GOLDEN_RATIO = 1.618

# ------------------ MEDIAPIPE INIT (OUTSIDE IF) ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# ------------------ HELPER FUNCTIONS ------------------
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def golden_similarity(ratio):
    return 1 - abs(GOLDEN_RATIO - ratio) / GOLDEN_RATIO

# ------------------ CAMERA INPUT ------------------
image = st.camera_input("Take a clear front-face photo")

# ------------------ IMAGE DEPENDENT LOGIC ------------------
if image is not None:
    st.write("Image captured successfully")

    # Convert image (already RGB)
    img = Image.open(image)
    img_np = np.array(img)

    # Run MediaPipe
    results = face_mesh.process(img_np)

    if results.multi_face_landmarks is None:
        st.error("No face detected. Please try again.")
    else:
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = img_np.shape
        points_468 = []

        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            z = lm.z
            points_468.append((x, y, z))

        # ------------------ MEASUREMENTS ------------------
        face_length = distance(points_468[10], points_468[152])
        face_width  = distance(points_468[33], points_468[263])
        mouth_width = distance(points_468[61], points_468[291])

        # ------------------ RATIOS ------------------
        ratio_face  = face_length / face_width
        ratio_mouth = mouth_width / face_width

        # ------------------ SCORE ------------------
        score_face  = golden_similarity(ratio_face)
        score_mouth = golden_similarity(ratio_mouth)

        final_score = (score_face * 0.7 + score_mouth * 0.3)
        beauty_score = round(final_score * 10, 2)

        # ------------------ OUTPUT ------------------
        st.success(f"Beauty Score: {beauty_score} / 10")
        st.image(img_np, caption="Captured Image", use_column_width=True)

        st.caption(
            "This score reflects similarity of facial proportions to the golden ratio, "
            "not personal worth or attractiveness."
        )
