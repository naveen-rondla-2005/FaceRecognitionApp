import streamlit as st
from face_ops import capture_images, verify_member_live, predict_age_live
from database import get_all_members, clear_database

if 'cv2_version' not in st.session_state:
    import cv2
    st.session_state['cv2_version'] = cv2.__version__
st.write(f"OpenCV version: {st.session_state.cv2_version}")

def main():
    st.title("Face Recognition App")
    menu = ["Add Member", "Check Member", "Check Age"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Add Member":
        st.subheader("Add a New Member")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, step=1)
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])

        if st.button("Start Camera"):
            if name.strip() and gender != "Select":
                count = capture_images(name, age, gender)
                if count > 0:
                    st.success(f"Captured and stored {count} face(s) for {name}.")
                else:
                    st.error("No face captured.")
            else:
                st.warning("Please enter a valid name and gender.")

    elif choice == "Check Member":
        st.subheader(choice)
        if st.button("Start Camera"):
            members = get_all_members()
            if not members:
                st.warning("No members in the database. Please add a member first.")
            else:
                st.info("Opening webcam. Press 'q' in the video window to stop.")
                verify_member_live()
    elif choice == "Check Age":
        st.subheader("Check Age")
        if st.button("Start Camera"):
            st.info("Opening webcam. Press 'q' in the video window to stop.")
            predict_age_live()

    st.subheader("Clear Database")
    if st.button("Clear"):
        clear_database()
        st.success("Database cleared successfully.")

if __name__ == "__main__":
    main()