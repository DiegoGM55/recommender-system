import pandas as pd
from typing import Tuple, Dict, Any

def load_and_prepare_data(path: str) -> pd.DataFrame:
    """
    Carrega o dataset limpo a partir de um arquivo Parquet e realiza
    a preparação inicial dos tipos de dados.
    """
    df = pd.read_parquet(path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    return df

def split_train_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o dataframe em conjuntos de treino e validação com base na
    última fatura de cada cliente.
    """
    last_invoice_dates = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    last_invoice_dates.rename(columns={'InvoiceDate': 'LastInvoiceDate'}, inplace=True)
    
    df_merged = pd.merge(df, last_invoice_dates, on='CustomerID')
    
    is_validation = df_merged['InvoiceDate'] == df_merged['LastInvoiceDate']
    
    train_df = df_merged[~is_validation].copy()
    validation_df = df_merged[is_validation].copy()
    
    return train_df, validation_df

def create_mappings(train_df: pd.DataFrame) -> Tuple[Dict[Any, int], Dict[Any, int], Dict[int, Any], Dict[int, Any]]:
    """
    Cria os dicionários de mapeamento para usuários e itens a partir dos
    dados de treino.
    """
    user_ids = train_df['CustomerID'].unique()
    user_map = {id: i for i, id in enumerate(user_ids)}
    inverse_user_map = {i: id for id, i in user_map.items()}
    
    item_ids = train_df['StockCode'].unique()
    item_map = {id: i for i, id in enumerate(item_ids)}
    inverse_item_map = {i: id for id, i in item_map.items()}
    
    return user_map, item_map, inverse_user_map, inverse_item_map