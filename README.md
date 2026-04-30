# RecipeChatbot 

RecipeChatbot is a capstone project that helps users find recipes through a chat interface. The app combines a React frontend, a FastAPI backend, Elasticsearch recipe search, and a local Ollama LLM workflow. Users can search by ingredients, dish names, nutrition goals, or exclusions, then select a recipe and ask follow-up questions about it.

## Features

- Natural-language recipe search, such as "high protein chicken under 500 calories" or "pasta without beef".
- Elasticsearch filtering by title, ingredients, excluded ingredients, and nutrition ranges.
- Conversational search refinement for adding constraints, asking for more options, or starting over.
- Recipe cards with titles and images in the chat UI.
- Follow-up Q&A for a selected recipe using Ollama and LangChain.
- Local recipe dataset with 2,520 scraped recipes.

## Tech Stack

### Backend

- Python
- FastAPI
- Elasticsearch
- LangChain
- Ollama with the `llama3.1` model
- Pydantic

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- Zustand
- shadcn/radix-style UI components

## Project Structure

```text
RecipeChatBot/
├── app.py                    # FastAPI app and API routes
├── llm_pre_es.py             # Search intent parsing, session state, and recipe selection logic
├── llm_handler.py            # Selected-recipe chat assistant
├── search.py                 # Elasticsearch recipe search tool
├── data_load_elastic.py      # Loads recipes into Elasticsearch
├── ingredient.py             # Scrapes ingredient/category links
├── recipe_url.py             # Scrapes recipe links from category pages
├── scrape_recipes.py         # Scrapes recipe JSON-LD data into JSONL
├── docker-compose.yml        # Local Elasticsearch service
├── requirements.txt          # Python dependencies
├── data/
│   ├── category_links.txt
│   ├── recipe_links.txt
│   └── recipes.jsonl
└── frontend/
    ├── src/
    ├── package.json
    └── vite.config.ts
```

## Prerequisites

Install these before running the project:

- Python 3.10 or newer
- Node.js and npm
- Docker Desktop
- Ollama

Pull the local LLM model used by the project:

```bash
ollama pull llama3.1
```

## Backend Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Elasticsearch:

```bash
docker compose up -d
```

Load the included recipe dataset into Elasticsearch:

```bash
python data_load_elastic.py
```

Start Ollama in a separate terminal:

```bash
ollama serve
```

Start the FastAPI backend:

```bash
fastapi dev app.py
```

The backend runs at:

```text
http://localhost:8000
```

You can check the backend with:

```text
http://localhost:8000/health
```

## Frontend Setup

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at the Vite URL shown in your terminal, usually:

```text
http://localhost:5173
```

The frontend sends requests to `/api/*`. Vite proxies those requests to the FastAPI backend at `http://localhost:8000`.

## UI Design

### Home Page:

<img width="1454" height="668" alt="image" src="https://github.com/user-attachments/assets/1ee0ece7-be5f-43d5-a34f-09a2e912cb29" />

### Chat Page:

<img width="1491" height="853" alt="image" src="https://github.com/user-attachments/assets/e7f3bf1a-0991-4d3c-aa5f-092fe1ae61c5" />



## How to Use

1. Open the frontend in your browser.
2. Enter a recipe request at the home page, for example:

```text
I want a low calorie chicken dinner with garlic
```

3. After the user submits a prompt, the website navigates to the chat page and displays three suggested recipes.
4. Pick one by saying something like:

```text
I'll take the first one
```

5. Ask follow-up questions about the selected recipe:

```text
What ingredients do I need?
How long does it take?
Can I make it without dairy?
```

You can also refine results before selecting a recipe:

```text
make it higher protein
no mushrooms
show me more
start over and search for pasta
```

To clear the recipe history, click the refresh button under the chatbot responses or refresh the whole page.

<img width="260" height="156" alt="image" src="https://github.com/user-attachments/assets/b94d2483-45d4-4514-a947-d86fd37c58db" />


## API Endpoints

### `GET /health`

Returns a simple health check:

```json
{
  "status": "ok"
}
```

### `POST /search`

Starts a new recipe search session.

Request:

```json
{
  "query": "high protein chicken under 500 calories"
}
```

Response:

```json
{
  "summary": "Recipe summary text...",
  "session_id": "generated-session-id",
  "recipe_cards": [
    {
      "id": "recipe source URL",
      "title": "Recipe title",
      "image_url": "image URL"
    }
  ]
}
```

### `POST /chat`

Continues an existing search session, selects a recipe, or chats about a selected recipe.

Request:

```json
{
  "session_id": "generated-session-id",
  "message": "I'll take the first one. What are the ingredients?",
  "recipe_text": null
}
```

Response:

```json
{
  "answer": "Assistant answer text...",
  "recipe_cards": [],
  "recipe_text": "Selected recipe JSON, when a recipe is selected"
}
```

### `DELETE /session/{session_id}`

Clears a backend search or chat session.

## Data Pipeline

The included dataset is already available in `data/recipes.jsonl`. If you want to regenerate it, run the scraper scripts in this order:

```bash
python ingredient.py
python recipe_url.py
python scrape_recipes.py
python data_load_elastic.py
```

The scraping scripts collect recipe category links, recipe page links, structured recipe data, nutrition data, instructions, images, and source URLs.

## Environment Notes

`search.py` reads `ES_URL` from the environment and defaults to:

```text
http://localhost:9200
```

Create a `.env` file in the project root if your Elasticsearch URL is different:

```text
ES_URL=http://localhost:9200
```


## Troubleshooting

- If recipe search returns no results, make sure Elasticsearch is running and `python data_load_elastic.py` completed successfully.
- If the assistant does not answer, make sure Ollama is running and the `llama3.1` model is installed.
- If the frontend shows request errors, make sure the FastAPI backend is running on port `8000`.
- If Docker reports port conflicts, stop the other service using port `9200` or change the Elasticsearch port mapping in `docker-compose.yml`.
