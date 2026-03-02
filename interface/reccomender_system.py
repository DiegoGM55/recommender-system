import streamlit as st
import pandas as pd


st.title("Recomendação de produtos")
k = 10
user_id = "XYZ"


st.write("#### Itens comprados pelo usuário {user_id}".format(user_id=user_id))


st.write("#### Top {k} recomendações para o usuário {user_id}".format(k=k, user_id=user_id))

st.sidebar.subheader("Baselines")
if st.sidebar.button("Popularidade"):
    st.write("Popularidade selecionado")
if st.sidebar.button("Knn"):
    st.write("Knn selecionado")
if st.sidebar.button("Apriori"):
    st.write("Apriori selecionado")

st.sidebar.subheader("Filtragem Colaborativa")
if st.sidebar.button("Als"):
    st.write("ALS selecionado")

st.sidebar.subheader("Filtragem Baseada em Conteúdo")
if st.sidebar.button("Tf-idf"):
    st.write("TF-IDF selecionado")

st.sidebar.subheader("Híbrido")
if st.sidebar.button("Híbrido"):
    st.write("Híbrido selecionado")

st.sidebar.header("Configurações do sistema")

def card_product(product_name:str, score: float, image_url: str):
    with st.container(border=True, width="stretch"):
        st.image(image_url, use_container_width=True, )
        st.write("##### {product_name}".format(product_name=product_name))
        st.write("Score: {score}".format(score=score))

def gallery_products(df: pd.DataFrame):
    cols = st.columns(3)  # Create 3 columns
    for index, row in df.iterrows():
        col = cols[index % 3]  # Cycle through columns
        with col:
            card_product(
                product_name=row['product_name'],
                score=row['score'],
                image_url=row['image_url']
            )

data = {
    'product_name': [
        'Produto A', 'Produto B', 'Produto C',
        'Produto D', 'Produto E', 'Produto F',
        'Produto G', 'Produto H', 'Produto I',
        'Produto J'
    ],
    'score': [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50],
    'image_url': [
        'https://placehold.co/400x400?text=Produto+A',
        'https://placehold.co/400x400?text=Produto+B',
        'https://placehold.co/400x400?text=Produto+C',
        'https://placehold.co/400x400?text=Produto+D',
        'https://placehold.co/400x400?text=Produto+E',
        'https://placehold.co/400x400?text=Produto+F',
        'https://placehold.co/400x400?text=Produto+G',
        'https://placehold.co/400x400?text=Produto+H',
        'https://placehold.co/400x400?text=Produto+I',
        'https://placehold.co/400x400?text=Produto+J'
    ]
}

df = pd.DataFrame(data)
gallery_products(df)

