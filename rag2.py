#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader

# Cache expensive models
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    generator = AutoModelForCausalLM.from_pretrained("gpt2")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    return tokenizer, generator, embedder

# Cache document processing
@st.cache_data
def process_document(uploaded_file):
    def split_into_chunks(text, chunk_size=600):
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > chunk_size and '.' in word:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    text = []
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        page_text = page.extract_text()
        cleaned = '\n'.join([line for line in page_text.split('\n') 
                           if not line.startswith('arXiv:')])
        text.append(cleaned)
    
    return split_into_chunks(' '.join(text))

# Main app
def main():
    st.title("📄 Research Paper QA with RAG")
    st.write("Upload a research paper and ask questions about its content")
    
    # Load models once
    tokenizer, generator, embedder = load_models()
    
    # File upload
    uploaded_file = st.file_uploader("Upload research paper (PDF)", type="pdf")
    
    if uploaded_file:
        # Process document and create index
        with st.spinner("Processing document..."):
            paper_chunks = process_document(uploaded_file)
            index = faiss.IndexFlatL2(768)
            embeddings = embedder.encode(paper_chunks)
            index.add(embeddings)
        
        # Query input
        query = st.text_input("Enter your question about the paper:", 
                            placeholder="What are the key findings...")
        
        if query:
            with st.spinner("Analyzing paper and generating answer..."):
                # Retrieve context
                query_embedding = embedder.encode([query])[0]
                _, indices = index.search(query_embedding.reshape(1, -1), 3)
                context = [paper_chunks[i] for i in indices[0]]
                
                # Generate answer
                prompt = f"Query: {query}\nContext: {' '.join(context)}\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
                outputs = generator.generate(
                    inputs.input_ids,
                    max_new_tokens=200,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1]
            
            # Display results
            st.subheader("Answer:")
            st.write(answer)
            
            with st.expander("See relevant passages from the paper"):
                for i, chunk in enumerate(context, 1):
                    st.write(f"**Passage {i}:**")
                    st.write(chunk)
                    st.divider()

if __name__ == "__main__":
    main()

