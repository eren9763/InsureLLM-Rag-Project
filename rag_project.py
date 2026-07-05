from pathlib import Path
import glob
import os

from chromadb import PersistentClient
from dotenv import load_dotenv
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge-base"

load_dotenv(override=True)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1-nano"
DB_NAME = str(BASE_DIR / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"  # not used due to high cost
hugging_face_embedding_model = "BAAI/bge-m3"
AVERAGE_CHUNK_SIZE = 500


def fetch_documents():
    documents = []
    all_folders = glob.glob(str(KNOWLEDGE_BASE_PATH / "*"))

    for folder_path in all_folders:
        doc_type = os.path.basename(folder_path)
        docs = glob.glob(os.path.join(folder_path, "*.md"))
        for doc in docs:
            with open(doc, "r", encoding="utf-8") as f:
                documents.append(
                    {
                        "doc_type": doc_type,
                        "page_content": f.read(),
                        "source": doc,
                    }
                )

    return documents


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    text: str = Field(
        description="A chunk of the original document, unchanged, roughly 500-600 characters"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "doc_type": document["doc_type"]}
        return Result(page_content=self.text, metadata=metadata)


class Chunks(BaseModel):
    chunks: list[Chunk]


class ReRanks(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


def make_prompt(document):
    how_many = (len(document["page_content"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["doc_type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["page_content"]}

Respond with the chunks.
"""


def make_message(document):
    return [{"role": "user", "content": make_prompt(document)}]


def process_document(document):
    messages = make_message(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def split_large_document(document, max_length=4000):
    text = document["page_content"]
    if len(text) <= max_length:
        return [document]

    parts = []
    for i in range(0, len(text), max_length):
        part_text = text[i : i + max_length]
        parts.append(
            {
                "doc_type": document["doc_type"],
                "page_content": part_text,
                "source": document["source"],
            }
        )
    return parts


def create_chunks(documents):
    chunks = []
    for doc in documents:
        for part in split_large_document(doc):
            chunks.extend(process_document(part))
    return chunks


emb = HuggingFaceEmbeddings(model_name=hugging_face_embedding_model)


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    vectors = emb.embed_documents(texts)
    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, metadatas=metas, documents=texts)
    return collection


chroma_client = PersistentClient(path=DB_NAME)
existing_collections = [c.name for c in chroma_client.list_collections()]

if collection_name in existing_collections:
    collection = chroma_client.get_collection(collection_name)
    print(f"Existing vectorstore loaded with {collection.count()} documents")
else:
    documents = fetch_documents()
    chunks = create_chunks(documents)
    collection = create_embeddings(chunks)


class Evaluation(BaseModel):
    faithfulness: float = Field(
        description="0 to 1 score: does the answer rely only on the given context, no hallucination"
    )
    relevancy: float = Field(description="0 to 1 score: how relevant is the answer to the question")
    reasoning: str = Field(description="Brief explanation of the scores")


def evaluate_answer(question, context, answer):
    eval_prompt = f"""
Evaluate the following RAG answer.

Question: {question}

Context provided to the model:
{context}

Generated answer:
{answer}

Score faithfulness (does the answer only use facts present in the context, no hallucination) from 0 to 1.
Score relevancy (does the answer actually address the question) from 0 to 1.
Give a brief one-sentence reasoning.
"""
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": eval_prompt}],
        response_format=Evaluation,
    )
    return Evaluation.model_validate_json(response.choices[0].message.content)


latest_eval = {"faithfulness": None, "relevancy": None, "reasoning": ""}


def answer_question(question, history):
    embed_question = emb.embed_query(question)
    results = collection.query(query_embeddings=[embed_question])
    retrieved_texts = results["documents"][0]
    context = "\n\n".join(retrieved_texts)
    sources = list(set(m["source"] for m in results["metadatas"][0]))
    system_prompt = f"""...Context:\n{context}"""
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": question}
    ]

    response = completion(model=MODEL, messages=messages, stream=True)

    stream_response = ""
    for chunk in response:
        word = chunk.choices[0].delta.content or ""
        stream_response += word
        yield stream_response

    source_names = [os.path.basename(s) for s in sources]
    final_with_sources = stream_response + "\n\n---\n **Kaynaklar:** " + ", ".join(source_names)
    yield final_with_sources

    eval_result = evaluate_answer(question, context, stream_response)
    latest_eval["faithfulness"] = eval_result.faithfulness
    latest_eval["relevancy"] = eval_result.relevancy
    latest_eval["reasoning"] = eval_result.reasoning


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
)

with gr.Blocks(
    theme=theme,
    title="RAG Assistant",
    fill_height=True,
) as demo:
    gr.Markdown(
        """
        # InsureLLM Assistant
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            chat = gr.ChatInterface(fn=answer_question, fill_height=True)

        with gr.Column(scale=1):
            with gr.Accordion("Evaluation Metrics", open=True):
                faithfulness_box = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    label="Faithfulness",
                    interactive=False,
                )

                relevancy_box = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    label="Answer Relevancy",
                    interactive=False,
                )

                reasoning_box = gr.Textbox(
                    label="Explanation",
                    lines=5,
                    interactive=False,
                )

            refresh_btn = gr.Button(
                "Refresh Metrics",
                variant="primary",
                size="lg",
            )

            def update_metrics():
                return (
                    latest_eval["faithfulness"],
                    latest_eval["relevancy"],
                    latest_eval["reasoning"],
                )

            refresh_btn.click(
                update_metrics,
                outputs=[
                    faithfulness_box,
                    relevancy_box,
                    reasoning_box,
                ],
            )

demo.launch(inbrowser=True)
