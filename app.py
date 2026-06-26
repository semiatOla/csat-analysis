import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# --- INITIALISATION DE NLTK ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- CONFIGURATION DES STOPWORDS ---
FR_STOPWORDS = set(stopwords.words('french'))
CUSTOM_STOPWORDS = [
    "nan", "ras", "r.a.s", "RAS", "Ras", "ok", "okay", "Ok", "Okay",
    "cool", "Cool", "daccord", "d", "accord", "merci", "aimerais",
    "cest", "c'est", "ca", "ça","ai", "a", "va", "deja", 
    "déjà", "cette", "être", "suis", "svp", "avoir", "alors", "vers", "puis",
    "faire", "quand", "peut", "non", "après", "car", "faut",
    "lors", "si", "sorte", "aller", "neant", "fait", 
    "veux", "veut", "leur", "leurs", "ya", "ce", "cet", "cette", "ces", "le", "la", "les", "un", "une", "des", 
    "du", "de", "la", "en", "et", "que", "qui", "dans", "pour", "par", 
    "sur", "avec", "dans", "se", "ses", "sa", "son", "notre", "votre",
    "leur", "leurs", "est", "sont", "ont", "a", "avez", "ils", "elles",
    "nous", "vous", "je", "tu", "il", "elle", "mais", "ou", "et", "donc", 
    "or", "ni", "car", "pas", "plus", "très", "tout", "tous", "faite", "faire",
    "pourquoi", "sans", "comme"
]
STOPWORDS = list(FR_STOPWORDS | set(CUSTOM_STOPWORDS))

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Analyse de Feedback CSAT",
    page_icon="🇫🇷",
    layout="wide"
)

# --- CHARGEMENT CACHÉ DES MODÈLES ---
@st.cache_resource
def load_sentiment_pipeline():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=-1
    )
    bert_pipeline = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return sentiment_pipeline, bert_pipeline

sentiment_model, embed_model = load_sentiment_pipeline()

# --- FONCTION DE CLUSTERING SÉMANTIQUE (SBERT) ---
def process_semantic_topics(df_pipeline, feedback_col, n_clusters=5):
    with st.spinner("Génération des représentations sémantiques (Embeddings SBERT)..."):
        texts = df_pipeline['Cleaned_Feedback'].tolist()
        embeddings = embed_model.encode(texts, show_progress_bar=False)

    with st.spinner("Regroupement sémantique des commentaires (K-Means)..."):
        n_clusters = min(n_clusters, len(df_pipeline))
        if n_clusters < 2:
            df_pipeline['Topic'] = 1
            df_pipeline['Topic_lab'] = "Thématique Générale"
            return df_pipeline

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)
        # On commence les thèmes à 1 au lieu de 0 pour l'affichage utilisateur
        df_pipeline['Topic'] = cluster_labels + 1

    with st.spinner("Extraction des mots-clés par thématique sémantique (c-TF-IDF)..."):
        # Regroupement des textes par cluster pour exécuter c-TF-IDF
        cluster_docs = df_pipeline.groupby('Topic')['Cleaned_Feedback'].apply(lambda x: " ".join(x)).tolist()
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=STOPWORDS)
        X_cluster = vectorizer.fit_transform(cluster_docs)
        
        terms = vectorizer.get_feature_names_out()
        topic_mapping = {}
        
        for cluster_id in range(n_clusters):
            row_scores = X_cluster[cluster_id].toarray()[0]
            top_indices = row_scores.argsort()[::-1][:4]
            top_words = [terms[idx] for idx in top_indices]
            
            # Association de l'identifiant réel du cluster (commence à 1) au libellé
            topic_mapping[cluster_id + 1] = f"Thème {cluster_id+1}: " + ", ".join(top_words)
        
        df_pipeline['Topic_lab'] = df_pipeline['Topic'].map(topic_mapping)
        
    return df_pipeline

# --- NETTOYAGE DU TEXTE EN FRANÇAIS ---
def clean_french_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9àâäéèêëîïôöùûüç\- \']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join([word for word in text.split() if word not in STOPWORDS])

# --- INITIALISATION DU SESSION STATE ---
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "processed_emb_df" not in st.session_state:
    st.session_state.processed_emb_df = None

# --- SIDEBAR INTERFACE ---
st.sidebar.title("Configuration des Données")
mode = st.sidebar.radio(
    "Choisir le flux de travail :",
    ["1. Traiter un nouveau fichier de feedback", "2. Charger un fichier CSV déjà traité"]
)

# --- WORKFLOW 1 : TRAITEMENT DE NOUVEAU FICHIER ---
if mode == "1. Traiter un nouveau fichier de feedback":
    st.header("⚙️ Étape 1 : Charger et traiter de nouveaux feedbacks")
    uploaded_file = st.file_uploader("Charger un fichier de feedback (CSV ou Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            df = None

        if df is not None:
            st.success(f"Fichier chargé avec succès ! Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
            st.write("Aperçu des premières lignes de données brutes :")
            st.dataframe(df.head(3))

            st.subheader("Identification des colonnes")
            col_options = df.columns.tolist()
            
            feedback_col = st.selectbox("Sélectionner la colonne des commentaires (Texte en français) :", col_options)
            csat_col = st.selectbox("Sélectionner la colonne du score CSAT (ex: note de 1 à 5) :", col_options)
            meta_cols = st.multiselect("Sélectionner les colonnes de métadonnées pour le filtrage (Optionnel) :", [c for c in col_options if c not in [feedback_col, csat_col]])

            nb_clusters = min(7, len(df))
            n_clusters = st.slider("Nombre de clusters / thèmes à générer :", min_value=1, max_value=50, value=nb_clusters)

            if st.button("🚀 Lancer le pipeline d'analyse"):
                if feedback_col == csat_col:
                    st.warning("La colonne des commentaires et celle du CSAT ne peuvent pas être identiques.")
                else:
                    cols_to_keep = [feedback_col, csat_col] + meta_cols
                    df_pipeline = df[cols_to_keep].copy()
                    
                    total_rows_before = len(df_pipeline)
                    empty_rows_before = df_pipeline[feedback_col].isna().sum()
                    avg_len_before = df_pipeline[feedback_col].dropna().apply(len).mean()

                    with st.spinner("Nettoyage des commentaires en français..."):
                        df_pipeline[feedback_col] = df_pipeline[feedback_col].fillna("")
                        df_pipeline['Cleaned_Feedback'] = df_pipeline[feedback_col].apply(clean_french_text)
                        df_pipeline = df_pipeline[df_pipeline['Cleaned_Feedback'] != ""]

                    total_rows_after = len(df_pipeline)
                    empty_rows_after = (df_pipeline['Cleaned_Feedback'] == "").sum()
                    avg_len_after = df_pipeline['Cleaned_Feedback'].apply(len).mean()

                    # Affichage des statistiques avant/après
                    st.subheader("📊 Statistiques de traitement du texte (Avant vs. Après)")
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.markdown("**Avant nettoyage**")
                        st.metric("Lignes totales", f"{total_rows_before}")
                        st.metric("Commentaires vides", f"{empty_rows_before}")
                        st.metric("Longueur moyenne", f"{avg_len_before:.1f} caractères")
                    with stat_col2:
                        st.markdown("**Après nettoyage**")
                        st.metric("Lignes restantes", f"{total_rows_after}")
                        st.metric("Commentaires rejetés", f"{empty_rows_after}")
                        st.metric("Longueur moyenne", f"{avg_len_after:.1f} caractères")

                    # Analyse des sentiments (Commune aux deux méthodes)
                    with st.spinner("Analyse sémantique des sentiments en cours (Modèle XLM-RoBERTa)..."):
                        try:
                            texts_list = df_pipeline['Cleaned_Feedback'].tolist()
                            raw_results = sentiment_model(texts_list, truncation=True, max_length=512, batch_size=8)
                            
                            sentiments = []
                            for res in raw_results:
                                label = res['label'].lower()
                                if '0' in label or 'neg' in label:
                                    sentiments.append("Négatif")
                                elif '1' in label or 'neu' in label:
                                    sentiments.append("Neutre")
                                elif '2' in label or 'pos' in label:
                                    sentiments.append("Positif")
                                else:
                                    sentiments.append(res['label'].capitalize())
                            
                            df_pipeline['Sentiment'] = sentiments
                        except Exception as nlp_err:
                            st.error(f"Échec de l'analyse des sentiments : {nlp_err}")
                            df_pipeline['Sentiment'] = "Neutre (Échec du modèle)"

                    # Création des deux structures de données distinctes
                    df_tfidf = df_pipeline.copy()
                    df_emb = df_pipeline.copy()

                    # 1. Méthode classique : TF-IDF
                    with st.spinner("Extraction des thèmes via TF-IDF..."):
                        vectorizer = TfidfVectorizer(max_features=1000, stop_words=STOPWORDS)
                        X = vectorizer.fit_transform(df_tfidf['Cleaned_Feedback'])
                        
                        if n_clusters >= 2:
                            kmeans_tf = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                            kmeans_tf.fit(X)
                            
                            terms = vectorizer.get_feature_names_out()
                            order_centroids = kmeans_tf.cluster_centers_.argsort()[:, ::-1]
                            
                            topic_mapping = {}
                            for i in range(n_clusters):
                                top_words = [terms[ind] for ind in order_centroids[i, :4]]
                                topic_mapping[i + 1] = f"Thème {i+1}: " + ", ".join(top_words)
                            
                            df_tfidf['Topic'] = kmeans_tf.labels_ + 1
                            df_tfidf['Topic_lab'] = df_tfidf['Topic'].map(topic_mapping)
                        else:
                            df_tfidf['Topic'] = 1
                            df_tfidf['Topic_lab'] = "Thématique Générale"

                    # Sauvegarde des résultats TF-IDF
                    st.session_state.processed_df = df_tfidf

                    # 2. Méthode sémantique : Embeddings SBERT
                    with st.spinner("Extraction des thèmes via Embeddings SBERT..."):
                        df_emb = process_semantic_topics(df_emb, feedback_col, n_clusters=n_clusters)
                        st.session_state.processed_emb_df = df_emb

                    st.success("🎉 Traitement terminé avec succès ! Consultez les résultats ci-dessous.")

    # Option de téléchargement rapide si les données existent
    if st.session_state.processed_df is not None:
        st.subheader("💾 Télécharger les résultats bruts traités")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_tf = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les résultats (Méthode TF-IDF)",
                data=csv_tf,
                file_name="feedback_traite_tfidf.csv",
                mime="text/csv"
            )
        with dl_col2:
            if st.session_state.processed_emb_df is not None:
                csv_emb = st.session_state.processed_emb_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats (Méthode Embeddings SBERT)",
                    data=csv_emb,
                    file_name="feedback_traite_embeddings.csv",
                    mime="text/csv"
                )

# --- WORKFLOW 2 : CHARGEMENT DE CSV EXISTANT ---
elif mode == "2. Charger un fichier CSV déjà traité":
    st.header("📂 Étape 1 : Charger un fichier d'analyse existant")
    uploaded_processed = st.file_uploader("Importer le fichier CSV précédemment traité", type=["csv"])
    
    if uploaded_processed is not None:
        try:
            df_loaded = pd.read_csv(uploaded_processed)
            expected = ['Sentiment', 'Topic', 'Topic_lab', 'Cleaned_Feedback']
            missing = [col for col in expected if col not in df_loaded.columns]
            
            if len(missing) == 0:
                # Dans ce mode, on copie les données chargées sur les deux variables pour pouvoir les analyser.
                st.session_state.processed_df = df_loaded
                st.session_state.processed_emb_df = df_loaded
                st.success("Fichier traité importé avec succès.")
            else:
                st.error(f"Le fichier importé n'a pas la structure requise. Colonnes manquantes : {missing}")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")


# --- DESSINER LE DASHBOARD (FONCTION GÉNÉRIQUE EN FRANÇAIS) ---
def render_analysis_dashboard(df_data, method_name, filename_suffix):
    working_df = df_data.copy()
    
    # Configuration des filtres
    st.subheader(f"⚙️ Paramètres des filtres ({method_name})")
    
    # Recherche dynamique des colonnes de métadonnées de filtrage possibles
    potential_filter_cols = ['Sentiment', 'Topic_lab']
    exclusion_list = ['Cleaned_Feedback', 'Sentiment', 'Topic', 'Topic_lab']
    
    # Ajouter les métadonnées de l'utilisateur si elles ont moins de 25 valeurs uniques
    for col in working_df.columns:
        if col not in exclusion_list and working_df[col].nunique() < 25:
            potential_filter_cols.append(col)
            
    col_sel, col_val = st.columns(2)
    with col_sel:
        selected_filter_col = st.selectbox(
            f"Colonne de filtrage principal ({method_name}) :", 
            potential_filter_cols, 
            key=f"filter_col_{filename_suffix}"
        )
    
    with col_val:
        unique_vals = sorted(working_df[selected_filter_col].dropna().unique().tolist())
        selected_vals = st.multiselect(
            "Valeurs à conserver :", 
            unique_vals, 
            default=unique_vals, 
            key=f"filter_val_{filename_suffix}"
        )
    
    # Application du filtre
    filtered_df = working_df[working_df[selected_filter_col].isin(selected_vals)]
    
    # Indicateurs de performance (KPIs)
    st.subheader("📈 Aperçu général des données filtrées")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    kpi1.metric("Nombre de feedbacks affichés", len(filtered_df))
    
    # Détection de la colonne CSAT
    possible_csat_cols = [
        c for c in filtered_df.columns 
        if c not in ['Cleaned_Feedback', 'Sentiment', 'Topic', 'Topic_lab'] 
        and pd.api.types.is_numeric_dtype(filtered_df[c])
    ]
    
    if len(possible_csat_cols) > 0:
        csat_column_to_use = possible_csat_cols[0]
        avg_csat_val = filtered_df[csat_column_to_use].mean()
        kpi2.metric(f"Score CSAT Moyen ({csat_column_to_use})", f"{avg_csat_val:.2f} / 5.0")
    else:
        kpi2.write("Aucune colonne de score CSAT numérique détectée.")
        csat_column_to_use = None

    if 'Sentiment' in filtered_df.columns:
        total_feedback = len(filtered_df)
        if total_feedback > 0:
            pos_ratio = (filtered_df['Sentiment'] == 'Positif').sum() / total_feedback * 100
            kpi3.metric("Taux de Sentiments Positifs", f"{pos_ratio:.1f}%")

    # Graphiques interactifs
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.markdown("#### Distribution des commentaires par Thème")
        if 'Topic_lab' in filtered_df.columns:
            topic_counts = filtered_df['Topic_lab'].value_counts().reset_index()
            topic_counts.columns = ['Thème', 'Nombre de commentaires']
            fig_topic = px.bar(
                topic_counts, 
                x='Nombre de commentaires', 
                y='Thème', 
                orientation='h', 
                title="Nombre de feedbacks par thème détecté"
            )
            st.plotly_chart(fig_topic, use_container_width=True, key=f"plot_topic_{filename_suffix}")
            
    with col_plot2:
        st.markdown("#### Répartition des sentiments par Thème")
        if 'Topic_lab' in filtered_df.columns and 'Sentiment' in filtered_df.columns:
            sentiment_topic = filtered_df.groupby(['Topic_lab', 'Sentiment']).size().reset_index(name='Nombre')
            sentiment_topic.columns = ['Thème', 'Sentiment', 'Nombre']
            fig_sent = px.bar(
                sentiment_topic, 
                x='Thème', 
                y='Nombre', 
                color='Sentiment', 
                title="Sentiments au sein de chaque thématique",
                color_discrete_map={'Positif': '#2ca02c', 'Neutre': '#bcbd22', 'Négatif': '#d62728'}
            )
            fig_sent.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_sent, use_container_width=True, key=f"plot_sent_{filename_suffix}")

    # Tableau croisé des statistiques par thème
    st.subheader("📋 Tableau croisé détaillé des thématiques")
    if 'Topic_lab' in filtered_df.columns:
        summary_group = filtered_df.groupby('Topic_lab')
        
        if csat_column_to_use:
            summary_df = summary_group.agg(
                Total_Commentaires=('Cleaned_Feedback', 'count'),
                CSAT_Moyen=(csat_column_to_use, 'mean')
            ).reset_index()
            summary_df['CSAT_Moyen'] = summary_df['CSAT_Moyen'].round(2)
        else:
            summary_df = summary_group.size().reset_index(name='Total_Commentaires')
            
        summary_df.columns = ["Thématique", "Total Commentaires", "CSAT Moyen"] if csat_column_to_use else ["Thématique", "Total Commentaires"]
        st.dataframe(summary_df, use_container_width=True)
        
        # Téléchargement de la table des statistiques
        stat_csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Télécharger la table de synthèse ({method_name})",
            data=stat_csv,
            file_name=f"synthese_themes_{filename_suffix}.csv",
            mime="text/csv",
            key=f"dl_stats_{filename_suffix}"
        )

    # Affichage des commentaires bruts filtrés
    st.subheader("🔍 Visionneuse détaillée des commentaires filtrés")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Téléchargement des commentaires filtrés
    slice_csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Télécharger ces commentaires filtrés (CSV)",
        data=slice_csv,
        file_name=f"export_feedbacks_filtres_{filename_suffix}.csv",
        mime="text/csv",
        key=f"dl_slice_{filename_suffix}"
    )


# --- RENTRÉE ET RENDU DES RÉSULTATS DANS LE TABLEAU DE BORD ---
if st.session_state.processed_df is not None:
    st.markdown("---")
    st.header("📊 Analyse interactive et exploration des résultats")
    
    # Séparation claire des résultats de TF-IDF et Embeddings sémantiques SBERT dans 2 onglets distincts
    tab_tfidf, tab_embeddings = st.tabs([
        "📝 Résultats : Méthode Lexicale (TF-IDF)", 
        "🧠 Résultats : Méthode Sémantique (Embeddings SBERT)"
    ])
    
    with tab_tfidf:
        st.markdown("### Modélisation de thèmes classique via TF-IDF")
        render_analysis_dashboard(
            st.session_state.processed_df, 
            "Méthode TF-IDF", 
            "tfidf"
        )
        
    with tab_embeddings:
        st.markdown("### Modélisation de thèmes sémantique via Embeddings SBERT")
        if st.session_state.processed_emb_df is not None:
            render_analysis_dashboard(
                st.session_state.processed_emb_df, 
                "Méthode SBERT", 
                "sbert"
            )
        else:
            st.info("Données sémantiques SBERT non disponibles. Veuillez retraiter un fichier à l'étape 1.")