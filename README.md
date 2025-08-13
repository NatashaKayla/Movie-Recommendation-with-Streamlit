# 🍿 Movie Recommendation with Streamlit (Content-Based)

Interactive **content-based** recommender for Netflix-style movie catalogs.
Pipeline: **preprocessing → EDA → TF-IDF vectorization → cosine similarity → Streamlit UI**.

> Repo includes a Jupyter notebook for data prep/EDA and a `Streamlit` app (`inferencee.py`) for serving recommendations.

---

## 🎯 Goals

* Clean the catalog and keep **movies only**
* Build robust **TF-IDF** features from textual metadata
* Compute **cosine similarity** for fast Top-N recommendations
* Ship a simple **Streamlit** interface for users

---

## 🗂️ Data

* Source: `netflix_titles.csv` (typical columns: `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`)

---

## 🧼 Data Preprocessing

* **Filter to movies:** drop all rows where `type == "TV Show"`.
* **Drop redundant columns:** remove `type` (no longer useful after filtering) and the identifier column (`show_id`).
* **Handle missing values:** fill textual nulls with **`"Unknown"`**.
* **Corpus construction:** concatenate relevant fields (`title`, `description`, `cast`, `director`, `listed_in`) into one `text` column.

### Anomaly Fixes

* Convert `date_added` to proper **datetime** dtype.
* Standardize unrated entries: replace `rating == "Unknown"` with **`"NR"`**.
* Clean mis-entered ratings: remove **`"74 min"`, `"66 min"`, `"84 min"`** from the **rating** column (these are durations, not age ratings).

---

## 🔍 Exploratory Data Analysis (EDA)

* Overall **distribution** of movies
* **Movies released over time** (by year)
* **Top 10 countries** by number of movies
* **Most common genres** (`listed_in`)
* **Distribution of movie durations** (minutes)
* **Top 10 directors** by title count
* **Distribution of movie ratings**

> Plots and counts are in the notebook.

---

## 🧠 Recommendation Method

### TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True
)

tfidf_matrix = tfidf.fit_transform(x['text'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

* Similarity: **cosine** (via `linear_kernel`)
* Retrieval: given a **title**, return **Top-N most similar** movies (excluding itself)

**Artifacts**

* `tfidf_matrix.pkl` — sparse TF-IDF matrix
* `smd.pkl` — metadata DataFrame aligned with TF-IDF rows

---

## 🖥️ Streamlit App (`inferencee.py`)

* Search/select a movie → display **Top-N** recommendations
* Shows title, director, cast, genres, release year, rating, overview
* Uses cached pickles for **fast** inference

---

## 📈 Evaluation

* **Qualitative** checks typical for content-based systems:

  * Face validity (themes/genres/crew overlap)
  * Quick user sanity checks in the UI

---

## 👥 Contributors

* **Natasha Kayla Cahyadi**
* **Jeremy Djohar Riyadi**
