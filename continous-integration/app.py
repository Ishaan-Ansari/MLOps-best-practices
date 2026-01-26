import streamlit as st 

st.title("Powered by Streamlit")

st.write("This is a simple Streamlit application.")

n = st.number_input("Enter a number", min_value=0, max_value=100, value=50)



# Calculate results
square = n ** 2
cube = n ** 3
fourth_power = n ** 4
fifth_power = n ** 5

# Display results
st.write(f"The square of {n} is: {square}")
st.write(f"The cube of {n} is: {cube}")
st.write(f"The fifth power of {n} is: {fifth_power}")
st.write(f"The fourth power of {n} is: {fourth_power}")
st.write("Thank you for using this application!")

