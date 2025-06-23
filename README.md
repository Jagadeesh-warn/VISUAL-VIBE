# 🎨 Visual Vibe: Content-Based Image Recommendation from Tumblr Likes

![Tumblr UI](./0c014692-7b0f-431f-8288-578f68f7f7c7.png)

Visual Vibe is a **content-based image recommendation system** that connects your aesthetic preferences with visually similar content from Tumblr. By analyzing your **liked posts** (especially image-based ones), the system recommends similar images using **computer vision techniques** like **SIFT**, **Bag of Visual Words (BoVW)**, and **K-Medoids clustering**.
<img width="1440" alt="image" src="https://github.com/user-attachments/assets/5c74ab44-6287-4b5a-963a-db23c15330c9" />


---

## 🚀 Features

- 🔐 OAuth2-based integration with Tumblr API
- 📸 Retrieval of user’s liked images
- 🧠 Feature extraction using SIFT descriptors
- 📦 Visual vocabulary creation using K-Medoids
- 📊 BoVW histogram representation for each image
- 🔍 Image similarity computation using Cosine Similarity
- 🤖 Intelligent recommendation of Tumblr content

---

## 🧰 Tech Stack

- Python (OpenCV, NumPy, Scikit-learn)
- Tumblr API + OAuth2
- SIFT for keypoint descriptors
- K-Medoids clustering (unsupervised learning)
- Flask (for optional web integration)

---

## 📂 Project Structure

    .
    ├── app.py               # Web interface (Flask)
    ├── main.py              # CLI for pipeline execution
    ├── oauth2.py            # Handles Tumblr OAuth login
    ├── tumblr_data.py       # Fetches liked images using Tumblr API
    ├── bovw.py              # Bag of Visual Words feature extraction
    ├── kmedoids.py          # Custom K-Medoids clustering
    ├── cbir.py              # Content-Based Image Retrieval engine
    ├── templates/           # HTML templates for result display
    ├── liked_photos/        # Downloaded liked images
    ├── candidate_photos/    # Optional: comparison image set
    └── README.md

---

## 🛠️ Setup & Installation

### 1. Clone the repo
    git clone https://github.com/harshinithangavel/visual-vibe.git
    cd visual-vibe

### 2. Create a virtual environment
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install dependencies

### 4. Configure Tumblr API keys

Generate OAuth credentials from https://www.tumblr.com/oauth/apps

Update `oauth2.py` with:
    client_id = 'your-client-id'
    client_secret = 'your-client-secret'

---

## ▶️ Running the Project

### Option 1: Command-Line Interface

    python main.py

### Option 2: Flask Web App

    python app.py

Then visit `http://127.0.0.1:5000/`

---

## 📸 Example Usage Flow

1. OAuth login to Tumblr
2. Fetch your liked images
3. A query image is analyzed
4. Compared using BoVW histogram
5. Visually similar Tumblr posts are recommended

---

## 📦 Key Concepts Used

| Stage               | Method Used                  |
|---------------------|------------------------------|
| Feature Extraction  | SIFT                         |
| Clustering          | K-Medoids                    |
| Representation      | Bag of Visual Words (BoVW)   |
| Similarity          | Cosine Similarity            |
| Retrieval           | Top-k ranking                |

---

## 📝 License

MIT License
