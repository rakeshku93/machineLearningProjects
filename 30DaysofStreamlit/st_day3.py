"st.button allows the display of a button widget."

import streamlit as st

# to add header to the web page
st.title("Demo App for testing st.button :smile:")

st.header("button-1")
# passing label argument in `st.button`
if st.button(label="Say Hello!", help="Optional mssg displayed when button is hover over."):
    # print the below text mssg using `st.write()`
    st.write("Why hello there")
    
else:
    st.write("Goodbye")
    

st.title("button-2")    
result = st.button("Click here")
st.write(result)
if result:
    st.write(":smile:")
else:
    pass
