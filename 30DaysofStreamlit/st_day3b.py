"Learn how to combine button, checkbox and radio button"

import streamlit as st

st.title("Radio Buttons, Checkboxes and Buttons")

page_names = ["Checkbox", "Button"]
page = st.radio("**Navigation**", page_names)

st.write("**The `st.radio` returns:**", page)

if page == "Checkbox":
    st.subheader("Welcome to the Checkbox page!")
    st.write("Nice to see you! :wave:")
    check = st.checkbox("Click here")
    st.write("State of the checkbox:", check)
    
    if check:
        st.write(":smile:"*12)
        
else:
    st.subheader("Welcome to the Button page!")
    st.write("Pleasure meeting you :thumbsup:")
    button = st.button("Click here")
    st.write("Status of the button:", button)
    if button:
        st.write(":smile:")
    