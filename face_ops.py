import cv2
import face_recognition
import os
from database import add_member, get_all_members
import streamlit as st

def capture_images(name, age, gender):
    cap = cv2.VideoCapture(0)
    count = 0
    img_dir = "data"
    
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    stframe = st.empty()  # Placeholder for live video frames

    while count < 1:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if face_encodings:
                add_member(name, age, gender, face_encodings[0])
                img_path = os.path.join(img_dir, f"{name}.jpg")
                cv2.imwrite(img_path, frame)
                count += 1
        
        # Show live video frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Capturing Face...', use_column_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return count

def verify_member_live():
    members = get_all_members()
    if not members:
        return

    known_encodings = [member["face_encoding"] for member in members.values()]
    member_names = list(members.keys())

    # Initialize counters for metrics
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    cap = cv2.VideoCapture(0)

    # Streamlit placeholders for metrics display
    stframe = st.empty()
    metrics_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_names = []

        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            top, right, bottom, left = face_locations[i]

            if True in matches:
                match_index = matches.index(True)
                name = member_names[match_index]
                detected_names.append(name)
                member_data = members[name]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                info_text = f"{name}, {member_data['age']}"
                cv2.putText(frame, info_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Count true positive for known member detected
                tp += 1
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Count false positive for unknown face detected as known
                fp += 1

        # For simplicity, assume all known members not detected in this frame are false negatives
        fn += max(0, len(member_names) - len(detected_names))

        # True negatives are hard to define in this live context; set to 0 for now
        tn = 0

        # Calculate precision, accuracy, f1score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        f1score = 2 * (precision * accuracy) / (precision + accuracy) if (precision + accuracy) > 0 else 0

        # Display metrics live
        metrics_placeholder.markdown(f"""
        **Precision:** {precision:.2f}  
        **Accuracy:** {accuracy:.2f}  
        **F1 Score:** {f1score:.2f}
        """)

        # Show live video frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Verifying Member...', use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_age_live():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            age_gender_preds = face_recognition.face_encodings(rgb_frame, face_locations)
            for i, face_encoding in enumerate(age_gender_preds):
                age = "Unknown"
                gender = "Unknown"
                # The face_recognition library does not directly support age and gender prediction.
                # This is a placeholder for where the age and gender prediction logic would go.
                # For now, we will just display "Unknown" for age and gender.
                top, right, bottom, left = face_locations[i]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Gender: {gender}", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show live video frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Predicting Age...', use_column_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()