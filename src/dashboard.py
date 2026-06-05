
import os

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="HAM10000 — Classification Lésions Cutanées",
    page_icon="🔬",
    layout="wide",
)

# --- Palette des 7 classes ---
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASSES_FR = {
    "akiec": "Kératose actinique",
    "bcc":   "Carcinome basocellulaire",
    "bkl":   "Kératose bénigne",
    "df":    "Dermatofibrome",
    "mel":   "Mélanome",
    "nv":    "Nævus mélanocytaire",
    "vasc":  "Lésion vasculaire",
}
DANGER = {
    "akiec": "⚠️ Potentiellement précancéreux",
    "bcc":   "⚠️ Cancer (rarement métastatique)",
    "bkl":   "✅ Bénin",
    "df":    "✅ Bénin",
    "mel":   "🔴 CANCER — Danger vital",
    "nv":    "✅ Bénin (grain de beauté)",
    "vasc":  "✅ Bénin",
}
COULEURS = {
    "akiec": "#e74c3c", "bcc": "#e67e22", "bkl": "#f1c40f",
    "df": "#2ecc71", "mel": "#9b59b6", "nv": "#3498db", "vasc": "#1abc9c",
}


# --- Chargement du modèle ---
@st.cache_resource
def charger_modele():
    """Charge le meilleur modèle sauvegardé."""
    import tensorflow as tf
    import h5py
    import json
    import tempfile
    import shutil

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best_model.h5")
    model_path = os.path.normpath(model_path)

    def strip_quant_config(obj):
        """Supprime récursivement quantization_config (champ Keras 3 inconnu de tf.keras 2)."""
        if isinstance(obj, dict):
            return {k: strip_quant_config(v) for k, v in obj.items() if k != 'quantization_config'}
        elif isinstance(obj, list):
            return [strip_quant_config(item) for item in obj]
        return obj

    tmp_path = None
    try:
        # Copie temporaire du .h5 avec quantization_config retiré de la config
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()
        tmp_path = tmp.name
        shutil.copy2(model_path, tmp_path)
        with h5py.File(tmp_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config = json.loads(f.attrs['model_config'])
                f.attrs['model_config'] = json.dumps(strip_quant_config(config))
        model = tf.keras.models.load_model(tmp_path, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# --- SIDEBAR ---
st.sidebar.title("🔬 HAM10000")
st.sidebar.markdown("**Classification de lésions cutanées**")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "🔍 Prédiction", "📊 Comparaison ML vs DL", "ℹ️ À propos"],
)


# ================================================================
# PAGE ACCUEIL — Pyramide de Minto (conclusion d'abord)
# ================================================================
if page == "🏠 Accueil":
    st.title("🔬 Skin Cancer Classification — HAM10000")
    st.markdown("---")

    # MINTO — NIVEAU 1 : CONCLUSION (en premier)
    st.markdown("### Message clé")
    st.success(
        "**Le Transfer Learning (EfficientNetB0) atteint un F1-weighted de 0.7773, "
        "soit +17% par rapport au ML classique (0.6640).** Le recall du mélanome passe "
        "de 0.24 (ML) à 0.64 (Transfer Learning) — réduisant de 53% les cas non détectés."
    )

    # MINTO — NIVEAU 2 : ARGUMENTS
    st.markdown("### 📌 Arguments")
    col1, col2, col3 = st.columns(3)
    col1.metric("🥉 Meilleur F1 (ML)", "0.6640", help="LR L1 Lasso")
    col2.metric("🥈 Meilleur F1 (CNN)", "0.7098", delta="+6.9%", help="CNN v2 optimisé")
    col3.metric("🥇 Meilleur F1 (TL)", "0.7773", delta="+17.1%", help="EfficientNetB0")

    st.markdown("""
    | Critère | ML Classique | CNN Custom | Transfer Learning |
    |---------|:---:|:---:|:---:|
    | **Exploite la spatialité** | ❌ (flatten+PCA) | ✅ | ✅ |
    | **Features pré-apprises** | ❌ | ❌ | ✅ (ImageNet 14M) |
    | **Adapté petit dataset** | ✅ | ⚠️ | ✅ |
    | **Temps d'entraînement** | ~1 min (CPU) | ~10 min (GPU) | ~15 min (GPU) |
    """)

    # MINTO — NIVEAU 3 : DONNÉES DÉTAILLÉES
    with st.expander("📂 Données détaillées — Distribution du dataset"):
        st.markdown("""
        | Classe | Nom complet | Images | % | Dangerosité |
        |--------|-------------|:------:|:-:|-------------|
        | nv | Nævus | 5 403 | 72.3% | ✅ Bénin |
        | bkl | Kératose bénigne | 727 | 9.7% | ✅ Bénin |
        | mel | Mélanome | 614 | 8.2% | 🔴 Cancer |
        | bcc | Carcinome baso. | 327 | 4.4% | ⚠️ Cancer |
        | akiec | Kératose actinique | 228 | 3.1% | ⚠️ Précancéreux |
        | vasc | Lésion vasculaire | 98 | 1.3% | ✅ Bénin |
        | df | Dermatofibrome | 73 | 1.0% | ✅ Bénin |
        """)

    # OIA — appliqué au résumé
    st.markdown("### 🔬 Analyse OIA")
    st.info(
        "**Observation :** Le ML classique atteint un recall de 0.24 pour le mélanome — "
        "76% des cas seraient non détectés. Le Transfer Learning porte ce recall à 0.64.\n\n"
        "**Insight :** Le ML écrase les images en vecteurs plats (PCA), détruisant les motifs "
        "visuels (bords, textures, asymétrie) que les dermatologues utilisent pour diagnostiquer. "
        "Le CNN préserve la spatialité (+158% recall mel) et le Transfer Learning exploite "
        "14M d'images ImageNet pour compenser le petit dataset (7470 images).\n\n"
        "**Action :** Déployer EfficientNetB0 fine-tuné (F1=0.7773). "
        "Prochaine étape : augmenter les classes minoritaires (df: 73, vasc: 98) "
        "et optimiser le seuil de décision pour viser un recall mel > 90%."
    )


# ================================================================
# PAGE PRÉDICTION — Format OIA
# ================================================================
elif page == "🔍 Prédiction":
    st.title("🔍 Prédiction de lésion cutanée")
    st.markdown("---")

    model, model_err = charger_modele()

    uploaded = st.file_uploader(
        "📤 Uploadez une image de lésion cutanée (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Image uploadée", use_container_width=True)

        if model is not None:
            # Preprocessing — le modèle intègre sa propre normalisation (Rescaling 1/255)
            img_resized = img.resize((224, 224))
            arr = np.array(img_resized, dtype=np.float32)
            arr = np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3), valeurs [0, 255]

            # Prédiction
            probas = model.predict(arr, verbose=0)[0]
            classe_idx = np.argmax(probas)
            classe_nom = CLASSES[classe_idx]
            confiance = probas[classe_idx] * 100

            with col2:
                # OIA — OBSERVATION
                st.markdown("#### 🔎 Observation")
                st.markdown(
                    f"La lésion est classée comme **{classe_nom}** "
                    f"({CLASSES_FR[classe_nom]}) avec une confiance de **{confiance:.1f}%**."
                )

                # OIA — INSIGHT
                st.markdown("#### 💡 Insight")
                st.markdown(f"**Dangerosité :** {DANGER[classe_nom]}")
                if classe_nom == "mel":
                    st.error(
                        "Le mélanome est le cancer de la peau le plus dangereux. "
                        "Un diagnostic précoce est crucial pour la survie du patient."
                    )
                elif classe_nom in ("bcc", "akiec"):
                    st.warning(
                        f"La classe {classe_nom} ({CLASSES_FR[classe_nom]}) est potentiellement maligne. "
                        "Un suivi médical est recommandé."
                    )
                else:
                    st.success(
                        f"La classe {classe_nom} ({CLASSES_FR[classe_nom]}) est généralement bénigne."
                    )

                # OIA — ACTION
                st.markdown("#### 🎯 Action recommandée")
                mel_proba = probas[CLASSES.index("mel")] * 100
                bcc_proba = probas[CLASSES.index("bcc")] * 100
                if mel_proba > 30 or classe_nom == "mel":
                    st.error(
                        f"🩺 **Consultation dermatologique urgente recommandée.** "
                        f"Probabilité mélanome : {mel_proba:.1f}%."
                    )
                elif bcc_proba > 30 or classe_nom in ("bcc", "akiec"):
                    st.warning("🩺 **Consultation dermatologique recommandée** dans les prochaines semaines.")
                else:
                    st.info("🩺 **Surveillance régulière.** Consultez en cas de changement de taille, forme ou couleur.")

            # Graphique des probabilités
            st.markdown("---")
            st.markdown("#### 📊 Probabilités par classe")
            df_prob = pd.DataFrame({
                "Classe": [f"{c} ({CLASSES_FR[c]})" for c in CLASSES],
                "Probabilité": probas * 100,
                "Couleur": [COULEURS[c] for c in CLASSES],
            }).sort_values("Probabilité", ascending=True)

            fig = px.bar(df_prob, x="Probabilité", y="Classe",
                         orientation="h", color="Classe",
                         color_discrete_map={f"{c} ({CLASSES_FR[c]})": COULEURS[c] for c in CLASSES})
            fig.update_layout(showlegend=False, xaxis_title="Probabilité (%)",
                              yaxis_title="", height=350)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"⚠️ Modèle non chargé : {model_err}")

    # Disclaimer
    st.markdown("---")
    st.warning(
        "⚠️ **Cet outil est à usage éducatif uniquement.** "
        "Il ne remplace en aucun cas l'avis d'un dermatologue qualifié. "
        "En cas de doute, consultez un professionnel de santé."
    )


# ================================================================
# PAGE COMPARAISON ML vs DL — Format OIA
# ================================================================
elif page == "📊 Comparaison ML vs DL":
    st.title("📊 Comparaison ML classique vs Deep Learning")
    st.markdown("---")

    # OIA — OBSERVATION
    st.markdown("### 🔎 Observation")
    st.markdown(
        "Le tableau ci-dessous compare les performances des approches ML classique, "
        "CNN custom et Transfer Learning sur le test set (1494 images, 7 classes)."
    )

    df_comp = pd.DataFrame({
        "Modèle":       ["LR L1 (Lasso)", "KNN", "Random Forest", "CNN v2 (optimisé)", "EfficientNetB0"],
        "F1-weighted":  [0.6640, 0.6580, 0.6069, 0.7098, 0.7773],
        "Accuracy":     [0.6533, 0.7256, 0.7229, 0.6693, 0.7497],
        "Recall mel":   [0.24,   "—",   "—",   0.62, 0.64],
        "Type":         ["ML", "ML", "ML", "DL", "DL (Transfer)"],
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # OIA — INSIGHT
    st.markdown("### 💡 Insight")
    st.info(
        "**Le CNN surpasse le ML classique** car il préserve les relations spatiales dans les images. "
        "Le ML classique (PCA + flatten) perd l'information de voisinage entre pixels, "
        "qui est essentielle pour distinguer les motifs visuels des lésions.\n\n"
        "**Le Transfer Learning va encore plus loin** en réutilisant des features visuelles "
        "pré-apprises sur 14 millions d'images ImageNet. Ces features de bas niveau "
        "(bords, textures, gradients) sont universelles et directement utiles pour la dermatologie."
    )

    # OIA — ACTION
    st.markdown("### 🎯 Action")
    st.success(
        "**Pour un déploiement en production :** EfficientNetB0 fine-tuné est recommandé.\n\n"
        "**Prochaines étapes :**\n"
        "- Augmenter le dataset des classes minoritaires (df: 73, vasc: 98 images)\n"
        "- Tester d'autres architectures (ResNet50, DenseNet121)\n"
        "- Optimiser le seuil de décision pour le mélanome (coût asymétrique FN >> FP)"
    )


# ================================================================
# PAGE À PROPOS
# ================================================================
elif page == "ℹ️ À propos":
    st.title("ℹ️ À propos du projet")
    st.markdown("---")

    st.markdown("""
    ### 📋 Description
    Projet de classification de lésions cutanées en **7 classes diagnostiques**
    à partir du dataset [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
    (~10 000 images dermatoscopiques).

    ### 👥 Auteurs
    **FREDERIC, GYLLS, NARCISSE**

    ### 🏗️ Architecture du projet
    | Jalon | Contenu |
    |-------|---------|
    | 1-2 | Data Wrangling + EDA |
    | 3-4 | ML Classique (baseline) + Évaluation |
    | 5-6 | CNN Custom + Optimisation |
    | 7 | Comparaison ML vs DL |
    | 8 | Transfer Learning (EfficientNetB0) |
    | 9 | Dashboard Streamlit (OIA + Minto) |

    ### 📐 Frameworks de communication
    - **Pyramide de Minto** : conclusion d'abord → arguments → données détaillées
    - **Framework OIA** : Observation → Insight → Action (chaque résultat répond au "So What?")

   
    """)
