import streamlit as st
import datetime

# Function to calculate the URL for a given week
def get_weekly_url(start_date, week_increment):
    target_date = start_date + datetime.timedelta(weeks=week_increment)
    formatted_date = target_date.strftime('%Y%m%d')
    url = f"https://www.nejm.org/image-challenge?ci={formatted_date}"
    return url

# Streamlit app
def main():
    st.title("NEJM Image Challenge Viewer")

    # Starting date: October 13, 2005
    start_date = datetime.datetime(2005, 10, 13)

    # Session state to keep track of the week increment
    if 'week_increment' not in st.session_state:
        st.session_state.week_increment = 0

    # Display the current date
    current_date = start_date + datetime.timedelta(weeks=st.session_state.week_increment)
    st.write(f"### Image ID: {st.session_state.week_increment + 1}")
    st.write(f"### Date: {current_date.strftime('%B %d, %Y')}")
    url = get_weekly_url(start_date, st.session_state.week_increment)
    st.markdown(f"[Go to Challenge]({url})")

    # Buttons for navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous"):
            if st.session_state.week_increment > 0:
                st.session_state.week_increment -= 1
    with col2:
        if st.button("Next"):
            if current_date < datetime.datetime.now() - datetime.timedelta(weeks=1):
                st.session_state.week_increment += 1

if __name__ == "__main__":
    main()

