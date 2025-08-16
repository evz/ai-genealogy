## Project Overview

For this project, I'd like to turn it into an application for AI-powered
genealogy digitization. To begin with, it should process Dutch family history
books using OCR and LLM technology to extract structured family data. I'd like
to have very simple user interfaces (no need for complex javascript) that
accept PDF scans of source material and extract localized plain text (in Dutch
and English for now). This source material should serve as corpora for a few
different things: 

* Extracting structured data about family relationships and turning that into a
  standard GEDCOM format.
* Using the same structured data about family relationships to generate wiki
  pages for each entity (Person, Family, Place, Event, etc).
* Allowing users to ask free form questions about the content of the corpora
* Generating genealogical research questions

To implement this, I'd like to use the following technologies:

* Django (for the web interface and data models)
* Celery (for background processing of things like the OCR task)
* PostgreSQL + pgvector (for storing the data and creating vectors for the RAG
  pipeline)
* Redis (as a Celery backend)
* Docker (to containerize the various pieces and make them simple to run
  locally for development as well as in a production setup)

I'd describe the pattern I'd like to follow to implement the parts that are
backend by an LLM (entity extraction, query retrieval, etc) like this:

It’s an anchor-aware, local RAG pipeline: you clean OCR, split into small
chunks, and attach deterministic anchors (genealogical IDs like II.1.a,
page/time, names/dates/places), then store text + embeddings + trigram +
phonetic keys in PostgresQL. At query time a hybrid retriever (vectors +
trigram/BM25 + Daitch–Mokotoff), fused with RRF, pulls candidates, expands to
adjacent chunks within the same anchor, and can optionally traverse a
lightweight relationship graph. The LLM (e.g., Aya via Ollama) answers over
that curated context, keeping same-name individuals separate and delivering
fast, grounded responses.

I have an Ollama server running on my local network at the-area.local

There are some very important lessons that I'd like to make sure you know about
from the last time I tried to implement something like this. They are in
markdown files called DESIGN_LESSONS_LEARNED.md and TESTING_LESSONS_LEARNED.md.
I don't want to have to end up scrapping everything and starting over again
with this attempt. Some context that might help is that app was a Flask app
while the one we'll be building here is Django. But I think the principles in
those files are still relevant 
