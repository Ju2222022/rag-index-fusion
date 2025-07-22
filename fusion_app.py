import streamlit as st
import faiss
import json
import os
import tempfile
import numpy as np

st.set_page_config(page_title="Fusion RAG", layout="centered")
st.title("🧠 Fusionner plusieurs index RAG (.faiss + .json)")

uploaded_faiss = st.file_uploader("🗂️ Chargez les fichiers `.faiss` à fusionner", accept_multiple_files=True, type="faiss")
uploaded_json = st.file_uploader("📄 Chargez les fichiers `.json` associés", accept_multiple_files=True, type="json")

if st.button("🚀 Fusionner"):
    if not uploaded_faiss or not uploaded_json:
        st.warning("Merci de charger au moins un fichier .faiss et un .json.")
    elif len(uploaded_faiss) != len(uploaded_json):
        st.error("Chaque fichier .faiss doit avoir son fichier .json associé (même nombre).")
    else:
        all_vectors = []
        all_chunks = []

        for faiss_file, json_file in zip(uploaded_faiss, uploaded_json):
            # Créer fichiers temporaires
            with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp_faiss, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp_json:
                tmp_faiss.write(faiss_file.read())
                tmp_faiss_path = tmp_faiss.name
                json_content = json_file.read().decode("utf-8")
                tmp_json.write(json_content)
                tmp_json_path = tmp_json.name

            # Charger index et chunks
            index = faiss.read_index(tmp_faiss_path)
            with open(tmp_json_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            vectors = index.reconstruct_n(0, index.ntotal)
            all_vectors.append(vectors)
            all_chunks.extend(chunks)

            os.remove(tmp_faiss_path)
            os.remove(tmp_json_path)

        # Fusion des vecteurs
        all_vectors_np = np.vstack(all_vectors)
        d = all_vectors_np.shape[1]
        merged_index = faiss.IndexFlatL2(d)
        merged_index.add(all_vectors_np)

        # Enregistrement
        faiss.write_index(merged_index, "final_index.faiss")
        with open("final_chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        st.success("🎉 Fusion terminée avec succès !")
        with open("final_index.faiss", "rb") as f:
            st.download_button("⬇️ Télécharger final_index.faiss", f, file_name="final_index.faiss")

        with open("final_chunks.json", "rb") as f:
            st.download_button("⬇️ Télécharger final_chunks.json", f, file_name="final_chunks.json")
