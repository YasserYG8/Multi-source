# Architecture Guide: Study Notes for the Demo

How the project works, file by file and concept by concept, to answer the teacher's *how*, *why* and *what if* questions.

**How to study:** read part A until you can draw the request diagram **from memory**. Then, for each file in part B, say out loud *what / how / why* in one sentence each. The jury mostly asks **why**.

## Contents

- [A. The big picture: follow one request](#a-the-big-picture-follow-one-request)
- [B. Each file: what, how, why](#b-each-file-what-how-why)
  - [B1. The ML part (`sentiment/`)](#b1-the-ml-part-sentiment)
  - [B2. The scripts](#b2-the-scripts)
  - [B3. The API part (`main.py`, `api/`)](#b3-the-api-part-mainpy-api)
  - [B4. The web page (`web/`)](#b4-the-web-page-web)
  - [B5. The tests (`tests/`)](#b5-the-tests-tests)
- [C. The core concepts](#c-the-core-concepts)
  - [C1. Machine learning](#c1-machine-learning)
  - [C2. OCR and ASR](#c2-ocr-and-asr)
  - [C3. Web and API](#c3-web-and-api)
  - [C4. Python and the server](#c4-python-and-the-server)
  - [C5. Tests](#c5-tests)
- [D. "What if…?" questions](#d-what-if-questions)

---

## A. The big picture: follow one request

The best way to understand the architecture is to follow **one image** from the browser to the answer.

```
 Browser (web/app.js)
   │  POST /predict/image   (multipart: fichier = I02.png)
   ▼
 main.py ── FastAPI app ── finds the route
   ▼
 api/routes.py      predict_from_image()
   ▼
 api/uploads.py     read_upload()  → reads the bytes, checks real type, size, image OK?
   │                 ✗ problem → 422
   ▼
 sentiment/pipeline.py   predict_image(bytes)        (runs in a thread)
   ▼
 sentiment/ocr.py        extract_text_from_image()   → "Colis ouvert et article manquant…"
   ▼
 sentiment/classifier.py predict_sentiment(text)
   │   vectorizer.pkl: text → numbers  (uses preprocessing.clean_text)
   │   model.pkl:      numbers → probabilities
   ▼
 {"source":"image", "texte_extrait":"…", "sentiment_predit":"negatif", "confiance":0.43}
   ▼
 api/schemas.py     PredictionResponse checks the answer
   ▼
 Browser shows the stamp NÉGATIF
```

**The 3 layers.** Each layer only talks to the layer below it.

| Layer | Folder | Job | Knows about HTTP? |
|---|---|---|---|
| Web | `web/` | Show buttons and results | Only as a client |
| API | `main.py`, `api/` | Receive requests, check inputs, send answers | **Yes** |
| ML | `sentiment/` | Turn input into text, then into a sentiment | **No** |

**Why this matters:** the `sentiment/` package works without FastAPI. `train.py` and `evaluate_pipelines.py` use it directly. If tomorrow you need a command-line tool or a different web framework, `sentiment/` does not change. This is called **separation of concerns**.

---

## B. Each file: what, how, why

### B1. The ML part (`sentiment/`)

#### `preprocessing.py`: cleaning the text

- **What:** `clean_text("Très DÉÇU !")` returns `"tres decu"`.
- **How:** 4 steps. Lowercase → remove accents → keep only letters a–z → remove stop words (`le`, `la`, `de`…).
- **Why remove accents?** OCR and Whisper sometimes lose them. "decu" and "déçu" must look the same to the model.
- **Why keep `ne`, `pas`, `sans`, `rien`?** They flip the meaning. "fonctionne" is good, "ne fonctionne pas" is bad.
- **Important:** this function is **saved inside `vectorizer.pkl`**. Training and the API always clean text the same way. If you rename this file, the `.pkl` can no longer load.

#### `classifier.py`: the model

- **What:** `predict_sentiment(text)` returns `("negatif", 0.43)`.
- **How:** it loads `vectorizer.pkl` and `model.pkl` **once** (`@lru_cache`). Then: text → vector → `predict_proba` → the class with the highest probability.
- **Why `EmptyTextError`?** If the OCR finds no text, we must not classify an empty string. The API turns this error into a 422.

#### `ocr.py`: image → text

- **How:** gray image → 2x bigger → auto-contrast → Tesseract in French.
- **Why gray + bigger?** Tesseract reads black text on a light background best, with letters about 20–30 pixels high.
- **Why the `tessdata/` folder?** The French language file had to live in the project, because `C:\Program Files` needs admin rights. Bonus: the project carries its own language file.
- **Why `TESSDATA_PREFIX` and not `--tessdata-dir`?** The folder name "ML Exam" has a **space**, which broke the command-line option.

#### `asr.py`: audio → text (ASR = Automatic Speech Recognition)

- **How:** faster-whisper, `small` model, CPU, `int8`, French, `vad_filter=True`.
- **Why `vad_filter`?** Without it, silent audio produced *"Sous-titres réalisés par la communauté d'Amara.org"*. Whisper learned from YouTube subtitles, so in silence it "remembers" subtitle credits.
- **Why `lru_cache`?** Loading the model takes ~15 s. We do it once.

#### `pipeline.py`: the cascade

- 3 functions, `predict_text`, `predict_image` and `predict_audio`, all return **the same dictionary**.
- **This is the heart of the project:** image and audio only add one step ("convert to text"), then everything goes through `build_result()`.

### B2. The scripts

| File | Job | When to run it |
|---|---|---|
| `train.py` | Train the model, test it on 18 texts, save `.pkl` and reports | When the data or the model changes |
| `evaluate_pipelines.py` | Run the full cascade on all 18 dataset files, compare with the references | To measure OCR and Whisper quality |

### B3. The API part (`main.py`, `api/`)

#### `main.py`: the app

- Creates `app = FastAPI(...)`. Uvicorn runs this object (`uvicorn main:app`).
- **`lifespan`:** code that runs at startup. It loads the `.pkl` files and starts loading Whisper **in a background thread**, so the server starts at once.
- **Exception handlers:** `EmptyTextError` and `InvalidAudioError` become **422**.
- **CORS:** allows a page from another address to call the API.

#### `api/schemas.py`: the contracts (Pydantic)

- `TextRequest`: the input must be `{"texte": "..."}`, with 1–5000 characters after removing spaces, and **no extra fields** (`extra="forbid"`).
- `PredictionResponse`: the output. `sentiment_predit` must be one of 3 values, and `confiance` must be between 0 and 1.
- **Why a response schema?** It guarantees the answer always has the same shape. It also generates the Swagger documentation.

#### `api/uploads.py`: file checks

- `detect_mime_type()` reads the **first bytes** of the file (see [C3](#c3-web-and-api)).
- `UploadedFile` (a Pydantic model) checks: not empty → known format → the right kind (image or audio) → not too big → the image really opens.
- `read_upload()` reads at most **10 MB + 1 byte**. A 1 GB file is never fully loaded into memory.

#### `api/routes.py`: the 4 routes

- `/predict/text` (JSON), `/predict/image` and `/predict/audio` (file), `/predict` (auto).
- **Why does `/predict` use a form, not JSON?** One route cannot easily accept both JSON and a file. A form can carry both a text field and a file field.
- **`run_in_threadpool`:** OCR and Whisper are slow, so they run in a separate thread (see [C4](#c4-python-and-the-server)).

#### `api/demo.py`: the test page

- Serves `web/index.html` at `/`, the CSS and JS at `/static/`, and the dataset files at `/demo/fichiers/`.

### B4. The web page (`web/`)

- `index.html` = structure, `style.css` = look, `app.js` = behavior.
- `app.js` uses `fetch()` to call the API, exactly like `curl` would. The page is just **another client** of the API.

### B5. The tests (`tests/`)

- `conftest.py`: **fixtures**, which are shared things like the test client and generated files.
- `test_api.py`: calls the API like a real client (**integration tests**).
- `test_sentiment.py`: tests small pieces alone (**unit tests**).

---

## C. The core concepts

### C1. Machine learning

#### TF-IDF (how text becomes numbers)

A model cannot read words. It needs numbers.

- **TF (term frequency):** how often a piece of text appears in **this** review.
- **IDF (inverse document frequency):** pieces that appear in **every** review (like "produit") get a **low** weight. Rare pieces (like "arnaque") get a **high** weight.
- **TF × IDF** gives a score for each piece. Each review becomes a long vector of scores.
- `sublinear_tf=True`: a piece seen 10 times does not count 10x more (it uses a logarithm).

#### Character n-grams (`char_wb`, 2 to 5)

- Instead of whole words, we cut words into small pieces: "livré" → ` l`, `li`, `liv`, `livr`, `ivré`…
- **Why?** "livré", "livraison" and "livrée" share `livr`, so the model sees they are related. It also handles OCR typos: "réceptionnee" still shares most pieces with "réceptionnée".
- **Result:** 89% with character pieces vs 83% with whole words.

#### Logistic Regression

- It learns **one weight per piece of text, per class**. For example, `arnaq` gets a big weight for negatif.
- For a review, it adds up the weights for each class, then **softmax** turns the 3 scores into **3 probabilities that sum to 1**.
- `confiance` = the biggest of the 3 probabilities.
- `C=10` controls **regularization** (how much the model avoids extreme weights). A bigger C means less regularization.

#### Evaluation

- **Train set** = the data the model learns from (24 reviews). **Test set** = new data to measure it (18 texts). **Never test on the training data**, because the model has memorized it.
- **Confusion matrix:** rows = the truth, columns = the prediction. The diagonal = correct answers.
- **Precision** for "negatif": of all the reviews I *called* negative, how many really were? **Recall:** of all the *really* negative reviews, how many did I find? **F1:** the balance of both.
- **Why not cross-validation?** With 24 examples, each fold has only 6 per class, and the scores were random (21–29%).

### C2. OCR and ASR

| | Tesseract (OCR) | Whisper (ASR) |
|---|---|---|
| Input | Pixels | Sound waves |
| Inside | An **LSTM** neural network that reads text lines | A **Transformer** (encoder + decoder) trained on 680,000 hours of audio |
| Steps | Find text lines → recognize letters | Audio → 16 kHz → **spectrogram** (an image of the sound) → encoder → decoder writes words |
| Language | `fra.traineddata` file | `language="fr"` |

- **beam_size=5:** Whisper keeps the 5 best sentence candidates while writing, then picks the best. It is more accurate than always taking the single best next word.
- **int8 (quantization):** the model's numbers are stored in 8 bits instead of 32. That means less memory and more speed, with almost the same quality.
- **faster-whisper:** the same Whisper model, run by a faster engine (CTranslate2).
- **VAD (Voice Activity Detection):** a small model that finds where someone is speaking, and cuts the silence.

### C3. Web and API

- **REST API:** you use URLs + HTTP methods (POST = "send data to process").
- **JSON vs multipart/form-data:** JSON is for text data. Multipart is for **files** (binary data). Putting a file inside JSON needs base64 encoding, which makes it 33% bigger.
- **HTTP status codes:**
  - **200** OK
  - **422** Unprocessable Entity: "I understood your request, but the content is wrong". FastAPI uses it for validation errors.
  - **400** Bad Request: the request itself is broken.
  - **500** Server error: a bug or a crash on our side.
- **MIME type:** a label like `image/png`. The client sends one, but it **can lie**.
- **Magic bytes:** the first bytes of a file reveal its real type. PNG files always start with `\x89PNG`, MP3 files with `ID3`. So we **check the content, not the label**.
- **Pydantic:** you describe the data with Python types, and Pydantic checks it automatically and writes clear errors.
- **OpenAPI / Swagger:** FastAPI reads your routes and schemas and **generates** the `/docs` page by itself.
- **CORS:** browsers block a page from calling an API on another address unless the API allows it.

### C4. Python and the server

- **async and the event loop:** FastAPI handles many requests on **one** thread by switching between them while they wait. A slow CPU task (OCR, Whisper) would **freeze** everything. `run_in_threadpool` moves it to another thread, so the server can answer other requests in the meantime.
- **lifespan:** code that runs at server start (and at server stop).
- **`lru_cache`:** remembers a function's result. We use it to load each model only once.
- **pickle / joblib:** saves Python objects to a file (`.pkl`) and loads them back.

### C5. Tests

- **Fixture:** something prepared once and shared, like the test client. `scope="session"` = prepared once for all tests.
- **TestClient:** sends fake HTTP requests to the app without a real server. With `with`, it also runs the lifespan.
- **parametrize:** one test function, run with many inputs (for example, 8 bad payloads).
- **marker `slow`** and **`skipif`:** group tests, or skip them when the dataset is missing.
- **Coverage (99%):** the percentage of code lines that the tests run.

---

## D. "What if…?" questions

| What if… | Answer |
|---|---|
| …someone uploads a 1 GB file? | We read only 10 MB + 1 byte, then refuse with 422. The memory stays safe. |
| …a PNG file is named `.txt`? | Accepted. We check the content, not the name. (There is a test for this.) |
| …an MP3 is sent to `/predict/image`? | 422 "Fichier image attendu". On `/predict`, it is detected as audio. |
| …the image has no text? | OCR returns "" → `EmptyTextError` → 422 "Aucun texte à analyser". |
| …the audio is silent? | The VAD removes everything → empty text → 422. |
| …the review is in English? | OCR and Whisper may still read it, but the model only learned French, so the result is unreliable. |
| …the review is mixed ("slow delivery but perfect product")? | The model must pick one class. This is a known limit. |
| …two users send audio at the same time? | Each request runs in its own thread. Whisper (1 worker) handles them **one after the other**, so the second one waits. Text requests stay fast. |
| …Tesseract is not installed? | `pytesseract` raises an error → **500**. It is not handled nicely. Be honest if the jury asks. |
| …`model.pkl` is missing? | The server **does not start**, because the lifespan fails. That is better than starting broken. |
| …I retrain the model? | `python train.py` overwrites the `.pkl` files. **Restart the server**, because `lru_cache` keeps the old model in memory. |
| …I add a 4th class ("mitigé")? | Add data, retrain, **and** update `Sentiment` in `schemas.py`. Otherwise the response check fails → 500. |
| …the confidence is very low (0.35)? | Today, we still return the class. Improvement idea: below a threshold, answer "incertain". |
| …the photo is taken by a phone in portrait? | `exif_transpose` turns it the right way. Tilted text still makes the OCR worse. |
| …the audio is longer than 30 s? | Whisper works in 30-second windows, so it still works (up to 10 MB). |
| …the internet is down during the demo? | Everything works. The font, the models and the language file are all local. |
