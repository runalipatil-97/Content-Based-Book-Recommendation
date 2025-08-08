import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset with cover image URLs
data = [
    [1,"The Lost Kingdom","A thrilling adventure of a young explorer who discovers a hidden kingdom","John Carter","https://covers.openlibrary.org/b/id/8231991-L.jpg"],
    [2,"Whispers of the Night","A mystery novel about strange whispers haunting a small town","Mary Adams","https://covers.openlibrary.org/b/id/12509454-L.jpg"],
    [3,"Ocean's Secret","A marine biologist uncovers secrets deep beneath the ocean","Peter White","https://covers.openlibrary.org/b/id/240727-L.jpg"],
    [4,"The Time Traveler's Diary","The journey of a man traveling across centuries through a magical diary","Linda Grey","https://covers.openlibrary.org/b/id/8235110-L.jpg"],
    [5,"Shadows in the Forest","A suspenseful story of hikers trapped in a haunted forest","David King","https://covers.openlibrary.org/b/id/10523319-L.jpg"],
    [6,"Journey to Mars","An astronaut’s first human mission to Mars turns into a survival challenge","Susan Lee","https://covers.openlibrary.org/b/id/9871534-L.jpg"],
    [7,"The Hidden Truth","An investigative journalist uncovers political corruption","Michael Brown","https://covers.openlibrary.org/b/id/12947646-L.jpg"],
    [8,"Under the Crimson Sky","A love story set in the midst of a civil war","Emily Stone","https://covers.openlibrary.org/b/id/9871691-L.jpg"],
    [9,"The Cursed Painting","A cursed painting brings misfortune to its owners","Anna Green","https://covers.openlibrary.org/b/id/9642388-L.jpg"],
    [10,"Code of the Future","A young programmer creates an AI that changes the world","Robert Hall","https://covers.openlibrary.org/b/id/11122294-L.jpg"],
    [11,"Desert Storm","A survival story of a pilot stranded in the desert","Richard West","https://covers.openlibrary.org/b/id/9871941-L.jpg"],
    [12,"Melody of the Heart","A gifted pianist navigates love and ambition","Sophia Taylor","https://covers.openlibrary.org/b/id/9871802-L.jpg"],
    [13,"The Last Heir","A royal family’s last surviving heir must reclaim the throne","Thomas Reed","https://covers.openlibrary.org/b/id/9871655-L.jpg"],
    [14,"City of Glass","A detective investigates crimes in a city made entirely of glass buildings","Olivia Turner","https://covers.openlibrary.org/b/id/10527677-L.jpg"],
    [15,"The Forgotten Island","Shipwreck survivors discover an island with dark secrets","Henry Clarke","https://covers.openlibrary.org/b/id/9871882-L.jpg"],
    [16,"Secrets of the Library","A librarian finds an ancient manuscript leading to treasure","Sarah Bell","https://covers.openlibrary.org/b/id/9871704-L.jpg"],
    [17,"The Quantum Code","A scientist unlocks a code that can alter reality","Daniel Scott","https://covers.openlibrary.org/b/id/9871591-L.jpg"],
    [18,"Footprints in the Snow","A cold-case murder investigation in a snowy mountain town","Laura Parker","https://covers.openlibrary.org/b/id/9871564-L.jpg"],
    [19,"The Eternal Flame","A fantasy tale of a warrior seeking a legendary flame to save her kingdom","Nathan Brooks","https://covers.openlibrary.org/b/id/9871495-L.jpg"],
    [20,"Voices from the Past","An archaeologist hears voices from ancient ruins","Grace Miller","https://covers.openlibrary.org/b/id/9871472-L.jpg"]
]

df = pd.DataFrame(data, columns=["book_id", "title", "summary", "author", "cover_url"])
df["content"] = df["summary"] + " " + df["author"]

# TF-IDF and cosine similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["content"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_books(title, num_recommendations=5):
    if title not in df["title"].values:
        return []
    idx = df.index[df["title"] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    recommendations = []
    for i, score in sim_scores:
        recommendations.append(df.iloc[i])
    return recommendations

# Streamlit UI
st.title("📚 Book Recommendation System with Covers")
selected_book = st.selectbox("Choose a book:", df["title"].values)

if st.button("Recommend"):
    recs = recommend_books(selected_book)
    if recs:
        st.subheader("You may also like:")
        for _, row in enumerate(recs):
            st.image(row["cover_url"], width=120)
            st.write(f"**{row['title']}** by {row['author']}")
            st.caption(row["summary"])
    else:
        st.write("No recommendations found.")
