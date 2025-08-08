# 📚 Content-Based Book Recommendation System

## Introduction
Finding books that match your interests can be difficult with so many options available. This project builds a content-based book recommendation system that suggests books similar to a selected title by analyzing book summaries and authors. It helps readers discover new books with themes and styles they like, improving the overall reading experience.

## How It Works
The recommendation engine uses the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert book summaries and author names into numerical vectors that capture the importance of words. Then, it calculates the cosine similarity between these vectors to measure how closely books are related in terms of content. Based on this similarity, the system recommends books that are most alike to the user's chosen book.

---

## Features
- Content-based filtering using TF-IDF on book summaries and authors  
- Cosine similarity to find books with similar content  
- Interactive Streamlit app with book cover images, authors, and summaries  
- Easy-to-use UI with dropdown selection and instant recommendations  

---

## Dataset
A sample dataset of 20 books is included with fields:  
`book_id`, `title`, `summary`, `author`, and `cover_url` for images.

---

## Installation & Setup

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/book-recommender.git
   cd book-recommender
